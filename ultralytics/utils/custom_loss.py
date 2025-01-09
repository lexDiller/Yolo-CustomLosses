import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, mining='batch_hard', batch_size=1024):
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.batch_size = batch_size

    def _batch_hard_triplet_loss(self, features, labels):
        # Преобразуем в float32 перед вычислением расстояний
        features = features.float()

        # Вычисляем попарные расстояния эффективно
        dist_mat = torch.cdist(features, features, p=2)

        # Создаем маски для позитивных и негативных пар
        labels_mat = labels.expand(len(labels), len(labels))
        pos_mask = labels_mat.eq(labels_mat.t())
        neg_mask = ~pos_mask

        # Исключаем диагональные элементы
        pos_mask.fill_diagonal_(0)

        # Находим самые сложные позитивные и негативные примеры
        pos_dist = torch.where(pos_mask, dist_mat, torch.tensor(float('inf'), device=features.device))
        neg_dist = torch.where(neg_mask, dist_mat, torch.tensor(float('inf'), device=features.device))

        # Находим hardest positive и hardest negative для каждого anchor
        hardest_pos_dist = torch.min(pos_dist, dim=1)[0]
        hardest_neg_dist = torch.min(neg_dist, dim=1)[0]

        # Вычисляем triplet loss только для валидных триплетов
        valid_mask = (hardest_pos_dist < float('inf')) & (hardest_neg_dist < float('inf'))
        if not valid_mask.any():
            return torch.tensor(0.0, device=features.device)

        # Вычисляем loss только для валидных триплетов
        triplet_loss = torch.clamp(hardest_pos_dist[valid_mask] - hardest_neg_dist[valid_mask] + self.margin, min=0.0)

        return triplet_loss.mean()

    def forward(self, features, labels):
        if len(features) <= 1:
            return torch.tensor(0.0, device=features.device)

        features = F.normalize(features.float(), p=2, dim=1)

        if len(features) > self.batch_size:
            total_loss = 0
            for idx in range(0, len(features), self.batch_size):
                end_idx = min(idx + self.batch_size, len(features))
                batch_features = features[idx:end_idx]
                batch_labels = labels[idx:end_idx]
                total_loss += self._batch_hard_triplet_loss(batch_features, batch_labels)
            return total_loss / ((len(features) + self.batch_size - 1) // self.batch_size)
        else:
            return self._batch_hard_triplet_loss(features, labels)

class TripletLoss_cls(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        # Преобразуем в float32 для вычислений
        features = features.float()
        features = F.normalize(features, p=2, dim=1)
        dist_mat = torch.cdist(features, features)

        n = labels.size(0)
        # Create masks for positive and negative pairs
        mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        mask_neg = ~mask_pos

        # Find hardest positive and negative pairs
        dist_ap = (dist_mat * mask_pos.float()).max(dim=1)[0]
        dist_an = (dist_mat * mask_neg.float() + mask_pos.float() * 1e9).min(dim=1)[0]

        # Calculate triplet loss
        loss = torch.clamp(dist_ap - dist_an + self.margin, min=0.0)

        # Average non-zero triplet losses
        valid_mask = loss > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=features.device)

        return loss