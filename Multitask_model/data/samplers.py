import numpy as np
from torch.utils.data import WeightedRandomSampler


def create_multitask_sampler(data_seg_orig, data_seg_inhouse, data_classif, num_samples):
    """
    Inspiré de MS-JEPA create_class_weighted_sampler (sampler: 'weighted').

    Principe : chaque TÂCHE pèse 1.0 au total (équilibre inter-tâches),
    ET au sein de la classification les positifs et les négatifs pèsent
    chacun 0.5 (équilibre intra-tâche pos/neg), sans lire aucun fichier NIfTI.

    Poids par sample :
      seg_orig    →  1 / N_seg_orig                      (uniforme dans la tâche)
      seg_inhouse →  1 / N_seg_inhouse                   (uniforme dans la tâche)
      cls positif →  1 / (2 × N_cls_pos)                 (total pos = 0.5)
      cls négatif →  1 / (2 × N_cls_neg)                 (total neg = 0.5)

    Vérification :
      total seg_orig    = N_seg_orig × 1/N_seg_orig        = 1.0
      total seg_inhouse = N_seg_inhouse × 1/N_seg_inhouse  = 1.0
      total cls         = N_pos × 1/(2N_pos) + N_neg × 1/(2N_neg) = 0.5 + 0.5 = 1.0

    → P(tâche) = 1/3 pour chaque tâche
    → P(positif | cls) = P(négatif | cls) = 0.5, quelle que soit la proportion réelle

    Args:
        data_seg_orig    : liste retournée par get_segmentation_data()
        data_seg_inhouse : liste retournée par get_inhouse_segmentation_data()
        data_classif     : liste retournée par get_classification_data()
                           (doit contenir la clé 'has_mask': True/False)
        num_samples      : nb de tirages par epoch (ex: 250 × batch_size)
    """
    n_seg_orig    = len(data_seg_orig)
    n_seg_inhouse = len(data_seg_inhouse)

    n_cls_pos = sum(1 for d in data_classif if d.get("has_mask", False))
    n_cls_neg = len(data_classif) - n_cls_pos

    assert n_cls_pos > 0, "Aucun sample de classification positif trouvé (has_mask=True)"
    assert n_cls_neg > 0, "Aucun sample de classification négatif trouvé (has_mask=False)"

    w_seg_orig    = 1.0 / n_seg_orig
    w_seg_inhouse = 1.0 / n_seg_inhouse
    w_cls_pos     = 1.0 / (2 * n_cls_pos)   # total positifs  = 0.5
    w_cls_neg     = 1.0 / (2 * n_cls_neg)   # total négatifs  = 0.5

    weights = (
        [w_seg_orig]    * n_seg_orig    +
        [w_seg_inhouse] * n_seg_inhouse +
        [w_cls_pos if d.get("has_mask", False) else w_cls_neg for d in data_classif]
    )

    print(
        f"create_multitask_sampler :\n"
        f"  seg_orig    : {n_seg_orig:4d} samples  poids={w_seg_orig:.5f}  total={w_seg_orig * n_seg_orig:.2f}\n"
        f"  seg_inhouse : {n_seg_inhouse:4d} samples  poids={w_seg_inhouse:.5f}  total={w_seg_inhouse * n_seg_inhouse:.2f}\n"
        f"  cls positif : {n_cls_pos:4d} samples  poids={w_cls_pos:.5f}  total={w_cls_pos * n_cls_pos:.2f}\n"
        f"  cls négatif : {n_cls_neg:4d} samples  poids={w_cls_neg:.5f}  total={w_cls_neg * n_cls_neg:.2f}\n"
        f"  → num_samples / epoch = {num_samples}"
    )

    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
