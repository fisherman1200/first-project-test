import time
import numpy as np
import torch
import os
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from datasets.topo_dataset import TopologyDataset
from datasets.alarm_dataset import AlarmDataset
from models.gnn_transformer import GNNTransformer  # 用于节点级别嵌入
from models.alarm_transformer import AlarmTransformer
from torch.nn import MultiheadAttention
from tqdm import tqdm
from utils.metrics_utils import MetricsLogger
from utils.visualize_embeddings import extract_embeddings, plot_tsne
from utils.model_utils import ModelSaver
from utils.path_utils import get_run_timestamp
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from trainers.losses import FocalLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.eval_plots import plot_confusion, plot_roc
from trainers.pretrain_gnn import pretrain_gnn
# from models.full_model import FullModel
from models.hetero_full_model import FullModel

torch.autograd.set_detect_anomaly(True)

def train_model(cfg):
    # 打印统一的运行时间戳，便于管理输出目录
    run_ts = get_run_timestamp()
    print("本次运行时间戳:", run_ts)
    # 启动gpu/cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("目前训练使用的Device:", device)
    # (1) 加载网络拓扑数据
    topo_ds = TopologyDataset(cfg.data.topo_path)
    x_dict, edge_index_dict, edge_attr_dict = topo_ds[0]
    # 把所有张量移到同一个device，确保后续计算在相同的device上
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}
    # node_ids 是真实节点的列表，对应的索引是 1..len(node_ids)，将0留给PAD
    node_map = {nid: idx + 1 for idx, nid in enumerate(topo_ds.node_ids)}

    # (2) 加载告警日志数据
    full_alarm_ds = AlarmDataset(cfg.data.alarm_path, node_map, max_len=cfg.data.max_len,
                                 window_milliseconds=cfg.data.window_milliseconds,
                                 step_milliseconds=cfg.data.step_milliseconds)
    # 划分训练集、验证集、测试集
    train_ds, val_ds, test_ds = random_split(full_alarm_ds,
                                             [cfg.data.num_train, cfg.data.num_val, cfg.data.num_test],
                                             generator=torch.Generator().manual_seed(42))

    # ----- 使用 WeightedRandomSampler 解决样本不均衡 -----
    # 统计训练集每条序列在 (is_root, is_true_fault) 维度上的组合
    train_combo = []
    for idx in train_ds.indices:
        sample = full_alarm_ds[idx]
        has_root = sample['is_root'].max().item()
        has_true = sample['is_true_fault'].max().item()
        # 0=无根告警，1=根告警但非真实故障，2=真实故障
        if has_root:
            combo = 2 if has_true else 1
        else:
            combo = 0
        train_combo.append(combo)

    # 计算每个组合类别的权重，数量越少权重越大
    class_count = np.bincount(train_combo, minlength=3)
    class_weights = 1.0 / (class_count + 1e-6)  # 避免除零
    sample_weights = [class_weights[c] for c in train_combo]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # 使用 sampler 对训练集中稀缺类别进行过采样
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False)

    # --- 第二阶段专用数据加载器：仅包含存在根告警的序列 ---
    def _filter_root(subset):
        root_idx = [i for i in subset.indices
                    if full_alarm_ds[i]['is_root'].max().item() == 1]
        return torch.utils.data.Subset(full_alarm_ds, root_idx)

    train_loader_stage2 = DataLoader(_filter_root(train_ds), batch_size=cfg.data.batch_size, shuffle=True)
    val_loader_stage2 = DataLoader(_filter_root(val_ds), batch_size=cfg.data.batch_size, shuffle=False)


    # (3) 初始化整体模型
    model = FullModel(
        cfg,
        topo_ds.feature_dim,
        full_alarm_ds[0]['text_feat'].shape[1]
    ).to(device)

    # print(model)

    if cfg.gnn.pretrained_path:
        # 文件存在 → 直接加载
        if os.path.exists(cfg.gnn.pretrained_path):
            print(f"加载预训练 GNN 参数: {cfg.gnn.pretrained_path}")
            model.gnn.load_state_dict(
                torch.load(cfg.gnn.pretrained_path, map_location=device)
            )
        else:
            # 文件不存在 → 先跑预训练
            print("未找到预训练 GNN 参数，开始无监督预训练任务……")
            pretrain_gnn(cfg)
            # 预训练结束后再检查并加载
            if os.path.exists(cfg.gnn.pretrained_path):
                print(f"预训练完成，加载生成的 GNN 参数: {cfg.gnn.pretrained_path}")
                model.gnn.load_state_dict(
                    torch.load(cfg.gnn.pretrained_path, map_location=device)
                )
            else:
                raise FileNotFoundError(
                    f"预训练后仍未生成文件: {cfg.gnn.pretrained_path}"
                )
    else:
        print("未配置预训练路径，跳过加载与预训练")

    # 预计算并缓存节点嵌入
    model.compute_node_embs(x_dict, edge_index_dict, edge_attr_dict)

    # (4) 定义训练优化目标
    optimizer = Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    optimizer_true = Adam(
        list(model.true_attention.parameters()) +
        list(model.shared_true.parameters())+
        list(model.head_true.parameters()),
        lr=cfg.training.lr_true,
        weight_decay=cfg.training.weight_decay
    )

    # 定义一个 scheduler，学习率调度：
    scheduler = StepLR(optimizer, step_size=cfg.training.lr_step_size, gamma=0.1)
    scheduler_true = StepLR(optimizer_true, step_size=cfg.training.lr_step_size, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 初始化训练早停(early stopping)变量
    best_val = float('inf')
    no_improve_times = 0

    # (5.1) 初始化混合精度训练(AMP)
    scaler = GradScaler()

    # (5.2)初始化 Logger，指定要记录的关键指标名
    logger = MetricsLogger(keys=[
        'train_loss', 'train_root_loss', 'train_true_loss',
        'val_loss', 'val_root_loss', 'val_true_loss',
        # 根告警分类指标
        'train_acc', 'train_precision', 'train_recall', 'train_f1',
        'val_acc', 'val_precision', 'val_recall', 'val_f1',
        # 真故障分类指标
        'train_true_acc', 'train_true_precision', 'train_true_recall', 'train_true_f1',
        'val_true_acc', 'val_true_precision', 'val_true_recall', 'val_true_f1'
    ])  # 可扩充更多指标

    # (5.3)初始化保存模型参数的方法
    saver = ModelSaver(base_dir='data/processed/model')

    # (5.4)实例化Focal Losses， 减少样本不平衡的影响
    focal_loss_root = FocalLoss(alpha=cfg.training.root_focal_alpha, gamma=cfg.training.root_focal_gamma)
    focal_loss_true = FocalLoss(alpha=cfg.training.true_focal_alpha, gamma=cfg.training.true_focal_gamma)

    # (6) 训练循环
    # --------- Helper Functions ---------
    def compute_loss(out_root, out_true, token_logits, batch, in_true_task=False):
        """计算损失函数"""
        root_label = batch['is_root'].max(dim=1).values  # [B]
        true_label = batch['is_true_fault'].max(dim=1).values  # [B]
        # root 使用 Focal Loss（或设置 class weight）, 在true任务训练时也要保持成果
        loss_root = focal_loss_root(out_root, root_label)
        # 序列级根因定位损失
        token_loss = focal_loss_root(
            token_logits.view(-1, 2),
            batch['is_root'].view(-1)
        )
        # root任务训练时(stage=1)不计算true的loss
        if in_true_task :
            mask = root_label == 1
            if mask.any():
                # 对 True-RCA 使用 Focal Loss，提升少数类的关注度
                loss_true = focal_loss_true(out_true[mask], true_label[mask])
            else:
                loss_true = out_root.new_tensor(0.0)
        else:
            loss_true = out_root.new_tensor(0.0)
        weighted_true = cfg.training.true_loss_weight * loss_true
        return loss_root + token_loss + weighted_true, loss_root, loss_true, token_loss

    def run_epoch(loader, train=True, in_true_task=False):
        """运行一个 Epoch，返回损失和评估指标"""
        model.train() if train else model.eval()
        total_loss = total_root = total_true = total_token = 0.0
        all_root_preds, all_root_labels = [], []       # 根告警指标
        all_true_preds, all_true_labels = [], []     # 真故障指标
        context = autocast('cuda', enabled=True) if train else torch.no_grad()
        for batch in tqdm(loader, desc='Train' if train else 'Val',ncols=100, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with context:
                out_root, out_true, token_logits = model(batch)
                loss, l_root, l_true, l_token = compute_loss(out_root, out_true, token_logits, batch, in_true_task)
            if train:
                # 6.6 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # 收集损失与预测结果
            total_loss += loss.item()
            total_root += l_root.item()
            total_true += l_true.item()
            total_token += l_token.item()

            # 先用临时变量存 batch 结果，别覆盖列表
            batch_root_preds = out_root.argmax(dim=1).detach().cpu().tolist()
            batch_root_labels = batch['is_root'].max(dim=1).values.detach().cpu().tolist()
            all_root_preds.extend(batch_root_preds)
            all_root_labels.extend(batch_root_labels)

            # 仅在存在根告警的样本上评估真故障分类
            mask = torch.tensor(batch_root_labels, dtype=torch.bool)
            if mask.any():
                t_pred = out_true.argmax(dim=1).detach().cpu()[mask].tolist()
                t_lab = batch['is_true_fault'].max(dim=1).values.detach().cpu()[mask].tolist()
                all_true_preds.extend(t_pred)
                all_true_labels.extend(t_lab)
        num_batches = len(loader)
        acc = accuracy_score(all_root_labels, all_root_preds)
        precision = precision_score(all_root_labels, all_root_preds, zero_division=0)
        recall = recall_score(all_root_labels, all_root_preds, zero_division=0)
        f1 = f1_score(all_root_labels, all_root_preds, zero_division=0)

        if all_true_labels:
            acc_t = accuracy_score(all_true_labels, all_true_preds)
            prec_t = precision_score(all_true_labels, all_true_preds, zero_division=0)
            rec_t = recall_score(all_true_labels, all_true_preds, zero_division=0)
            f1_t = f1_score(all_true_labels, all_true_preds, zero_division=0)
        else:
            acc_t = prec_t = rec_t = f1_t = 0.0

        return (
            total_loss / num_batches, total_root / num_batches, total_true / num_batches,
            acc, precision, recall, f1,
            acc_t, prec_t, rec_t, f1_t,
            total_token / num_batches
        )

    # 使用 tqdm 进度条显示每个 epoch 的平均损失
    train_start = time.time()  # 记录训练开始时间
    epoch_bar = tqdm(range(cfg.training.epochs), desc="Epoch", unit="epoch", ncols=100)
    for epoch in epoch_bar:
        is_true_task = epoch >= cfg.training.stage1_epochs
        if epoch == cfg.training.stage1_epochs and is_true_task:
            # 重新计数
            no_improve_times = 0
            best_val = float('inf')
            # 第二阶段：冻结除 true fault 头之外的模块
            for module in [model.gnn, model.alarm_transformer,
                           model.shared_root, model.head_root, model.token_root_head]:
                for p in module.parameters():
                    p.requires_grad = False
            optimizer = optimizer_true
            scheduler = scheduler_true

        # ---- Train ----
        train_used = train_loader_stage2 if is_true_task else train_loader
        (train_loss, train_root, train_true,
         train_acc, train_prec, train_rec, train_f1,
         train_true_acc, train_true_prec, train_true_rec, train_true_f1,
         train_token) = run_epoch(train_used, train=True, in_true_task=is_true_task)
        # ---- Validation ----
        val_used = train_loader_stage2 if is_true_task else train_loader
        (val_loss, val_root, val_true,
         val_acc, val_prec, val_rec, val_f1,
         val_true_acc, val_true_prec, val_true_rec, val_true_f1,
         val_token) = run_epoch(val_used, train=False, in_true_task=is_true_task)

        # 在进度条尾部显示当前 epoch 的各项损失<在每个 epoch 末尾，拼一个想要的输出
        post_str = (
            f"Epoch {epoch}: \n"
            f"Total_loss：train={train_loss:.4f}, val={val_loss:.4f}  |  "
            f"Root_loss：train={train_root:.4f}, val={val_root:.4f}  |  "
            f"True_loss：train={train_true:.4f}, val={val_true:.4f}\n"
            f"Root-acc：train={train_acc:.4f}, val={val_acc:.4f}  |  "
            f"Root-precision：train={train_prec:.4f}, val={val_prec:.4f}  |  "
            f"Root-recall：train={train_rec:.4f}, val={val_rec:.4f}  |  "
            f"Root-F1：train={train_f1:.4f}, val={val_f1:.4f}\n"
            f"True-acc：train={train_true_acc:.4f}, val={val_true_acc:.4f}  |  "
            f"True-precision：train={train_true_prec:.4f}, val={val_true_prec:.4f}  |  "
            f"True-recall：train={train_true_rec:.4f}, val={val_true_rec:.4f}  |  "
            f"True-F1：train={train_true_f1:.4f}, val={val_true_f1:.4f}"
        )
        tqdm.write(post_str)  # → 直接输出完整一行，不会被截断
        # 然后单独用 set_postfix_str：
        # epoch_bar.set_postfix_str(post_str)

        #  添加到 logger
        avg_metrics = {
            'train_loss': train_loss,
            'train_root_loss': train_root,
            'train_true_loss': train_true,
            'val_loss': val_loss,
            'val_root_loss': val_root,
            'val_true_loss': val_true,
            'train_acc': train_acc,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'train_f1': train_f1,
            'val_acc': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1,
            'train_true_acc': train_true_acc,
            'train_true_precision': train_true_prec,
            'train_true_recall': train_true_rec,
            'train_true_f1': train_true_f1,
            'val_true_acc': val_true_acc,
            'val_true_precision': val_true_prec,
            'val_true_recall': val_true_rec,
            'val_true_f1': val_true_f1,
        }
        logger.add(epoch, avg_metrics)

        # —— Scheduler & Early Stopping ——
        scheduler.step()
        monitored_val = val_true if is_true_task else val_root
        metric_name = "true" if is_true_task else "root"
        if monitored_val < best_val:
            best_val = monitored_val
            no_improve_times = 0
            saved = saver.save_best(model=model)
        else:
            no_improve_times += 1
            if no_improve_times >= cfg.training.early_stop_patience:
                epoch_bar.write(f"Early stopping at epoch {epoch} ({metric_name}-loss no improvement)")
                break

    elapsed = time.time() - train_start
    print(f"训练时间：{elapsed:.2f} 秒")

    # —— 训练 & 验证 结束后，加载最佳模型权重，再对 test_loader 评估一次 ——
    print("Loading best model weights and evaluating on TEST set…")
    model.load_state_dict(torch.load(saved['model']))
    model.eval()

    # 测试评估
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Number of test batches: {len(test_loader)}")
    (
        test_loss, _, _,
        test_acc, test_prec, test_rec, test_f1,
        test_true_acc, test_true_prec, test_true_rec, test_true_f1,
        _
    ) = run_epoch(test_loader, train=False, in_true_task=True)
    print(
        f"Final TEST Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f} | "
        f"TrueAcc: {test_true_acc:.4f}, TrueF1: {test_true_f1:.4f}"
    )
    # --- Confusion Matrix & ROC Curve ---
    def eval_with_probs(loader):
        """收集预测概率与标签"""
        root_probs_f = []
        root_labels_f = []
        true_probs_f = []
        true_labels_f = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out_root, out_true, _ = model(batch)
                prob_root = out_root.softmax(dim=1)[:, 1].cpu()
                label_root = batch['is_root'].max(dim=1).values.cpu()
                prob_true = out_true.softmax(dim=1)[:, 1].cpu()
                label_true = batch['is_true_fault'].max(dim=1).values.cpu()
                root_probs_f.extend(prob_root.tolist())
                root_labels_f.extend(label_root.tolist())
                mask = label_root == 1
                true_probs_f.extend(prob_true[mask].tolist())
                true_labels_f.extend(label_true[mask].tolist())
        return (root_labels_f, root_probs_f), (true_labels_f, true_probs_f)

    (root_labels, root_probs), (true_labels, true_probs) = eval_with_probs(train_loader)
    _, best_threshold = plot_roc(root_labels, root_probs, "root")
    root_preds_opt = [1 if p >= best_threshold else 0 for p in root_probs]
    plot_confusion(root_labels, root_preds_opt, ("Derived", "Root"), "root_confusion")
    if true_labels:
        _, best_threshold = plot_roc(true_labels, true_probs, "true")
        true_preds_opt = [1 if p >= best_threshold else 0 for p in true_probs]
        plot_confusion(true_labels, true_preds_opt, ("NonFault", "True"), "true_confusion")


    # —— 可视化 & 其他后处理 ——
    # utils-> plot_metrics: 生成metrics数据json
    logger.save(prefix='full_model')

    # utils-> visualize_embeddings 绘制聚类效果图
    # 提取 embeddings 和标签
    embs, is_root, is_true = extract_embeddings(cfg, model.gnn, model.alarm_transformer, topo_ds, full_alarm_ds)
    # 根源 vs 衍生
    plot_tsne(embs, is_root, 't-SNE: Root vs Derived', "tsne_root_vs_derived", names=('Derived', 'Root'))
    # 真故障根源 vs 非真故障根源
    mask_root = is_root == 1
    plot_tsne(embs[mask_root], is_true[mask_root], 't-SNE: True Fault among Root',
              "tsne_true_among_root", names=('Non-Fault', 'True Fault'))
