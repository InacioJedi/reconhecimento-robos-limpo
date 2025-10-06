# label_collisions.py

import os
import cv2

# ——————————————————————————————
# CONFIGURAÇÃO
# Pasta onde estão as imagens
IMG_DIR   = 'C:\\Users\\needc\\Downloads\\reconhecimento-robos-main\\reconhecimento-robos-main\\data\\images\\train'   # <-- atualize para o seu diretório
# Pasta onde serão salvos os labels
LABEL_DIR = 'C:\\Users\\needc\\Downloads\\reconhecimento-robos-main\\reconhecimento-robos-main\\data\\images\\labels'
# Classe para evento de colisão (apenas uma classe = 0)
CLASS_ID  = 0
# ——————————————————————————————

def normalize_bbox(x1, y1, x2, y2, w, h):
    # YOLO: cx, cy, w, h (tudo normalizado [0,1])
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh

def annotate():
    os.makedirs(LABEL_DIR, exist_ok=True)
    img_files = [f for f in os.listdir(IMG_DIR)
                 if f.lower().endswith(('.png','.jpg','.jpeg','bmp'))]
    img_files.sort()
    
    for img_name in img_files:
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f'⚠ Não foi possível ler {img_name}, pulando.')
            continue

        h, w = img.shape[:2]
        clone = img.copy()
        bboxes = []
        drawing = False
        ix, iy = -1, -1

        def mouse_cb(evt, x, y, flags, param):
            nonlocal ix, iy, drawing, img
            if evt == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif evt == cv2.EVENT_MOUSEMOVE and drawing:
                img = clone.copy()
                cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), 2)
            elif evt == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, y1, x2, y2 = ix, iy, x, y
                bboxes.append((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))
                img = clone.copy()
                for bb in bboxes:
                    cv2.rectangle(img, bb[:2], bb[2:], (0,255,0), 2)

        cv2.namedWindow('Annotate')
        cv2.setMouseCallback('Annotate', mouse_cb)

        print(f'Annotando: {img_name} (caixas atualmente: {len(bboxes)})')
        while True:
            cv2.imshow('Annotate', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):   # salvar anotações
                lbl_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + '.txt')
                with open(lbl_path, 'w') as f:
                    for bb in bboxes:
                        cx, cy, bw, bh = normalize_bbox(*bb, w, h)
                        f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                print(f'[S] Salvo em {lbl_path}')
                break
            elif key == ord('n'): # pular sem salvar
                print(f'[N] {img_name} pulada')
                break
            elif key == ord('q'): # sair
                print('Saindo...')
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

if __name__ == '__main__':
    annotate()
