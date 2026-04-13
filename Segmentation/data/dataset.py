import os



def get_data_files(img_dir, seg_dir):
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
    labels = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    return [{"image": img, "seg": lbl} for img, lbl in zip(images, labels)]


def get_data_files_2(data_dir):
    """
    statégie 1 : random select one label per patient

    """
    
    data_pairs = []
    #     # Liste tous les sous-dossiers (chaque dossier correspond à un patient/cas)
    patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
  

    for patient_dir in patient_dirs:
        image_file = os.path.join(patient_dir, 'image.nii.gz')
            
    #         # S'assure que le fichier image existe avant de continuer
        if os.path.exists(image_file):
    #             # Trouve tous les fichiers d'annotation dans le dossier du patient
            label_files = [
                    os.path.join(patient_dir, f) 
                    for f in os.listdir(patient_dir) 
                if f.startswith('label_annot_') and f.endswith('.nii.gz')
                ]
            #print (f"label files : {label_files}")
            
            # ajouter annoration random:
            if label_files:
                import random
                selected_label = random.choice(label_files)
                data_pairs.append({"image": image_file, "seg": selected_label})
                #print (f"selected label : {selected_label}")
            
    return data_pairs

import os

def get_data_files_3(data_dir):
    """
    
    stratégie 2 :
    Parcourt les sous-dossiers d'un répertoire donné pour trouver des paires image/segmentation.
    Chaque sous-dossier doit contenir un 'image.nii.gz' et un ou plusieurs 'label_annot_*.nii.gz'.
    Crée une paire pour chaque annotation trouvée.
    """
    data_pairs = []
    # Liste tous les sous-dossiers (chaque dossier correspond à un patient/cas)
    patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for patient_dir in patient_dirs:
        image_file = os.path.join(patient_dir, 'image.nii.gz')
        
        # S'assure que le fichier image existe avant de continuer
        if os.path.exists(image_file):
            # Trouve tous les fichiers d'annotation dans le dossier du patient
            label_files = [
                os.path.join(patient_dir, f) 
                for f in os.listdir(patient_dir) 
                if f.startswith('label_annot_') and f.endswith('.nii.gz')
            ]
            
            # Crée une paire {"image": ..., "seg": ...} pour chaque label trouvé
            for label_file in label_files:
                data_pairs.append({"image": image_file, "seg": label_file})
                
    return data_pairs