"""IFCNN inference on algae test data — for visual comparison with V5"""
import sys, torch, cv2, numpy as np
from pathlib import Path

IFCNN_DIR = Path('../tools/IFCNN/Code')
sys.path.insert(0, str(IFCNN_DIR))
from model import IFCNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_ifcnn():
    ckpt_path = 'F:/Commercial project/multi-focus/tools/IFCNN/Code/snapshots/IFCNN-MAX.pth'
    model = IFCNN(1)  # resnet=1 for MAX fusion
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model


def fuse_ifcnn(model, srcs_512):
    """IFCNN pairwise fusion: fuse(1,2)->3->4->5"""
    # srcs_512: list of (H, W, 3) uint8
    # Convert to tensors (1, 1, H, W) grayscale
    tensors = []
    for s in srcs_512:
        gray = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(DEVICE)
        tensors.append(t)

    with torch.no_grad():
        result = tensors[0]
        for i in range(1, len(tensors)):
            result = model(result, tensors[i])  # pair-wise sequential

    fused = result[0, 0].cpu().numpy()
    fused = (np.clip(fused, 0, 1) * 255).astype(np.uint8)
    return fused


if __name__ == '__main__':
    model = load_ifcnn()
    print(f'IFCNN loaded on {DEVICE}')
    # Quick test
    import cv2
    srcs = []
    for i in range(1, 6):
        s = cv2.cvtColor(cv2.imread(f'all_data/split_data/test/group_003/img_{i}.png'), cv2.COLOR_BGR2RGB)
        srcs.append(cv2.resize(s, (512, 512)))
    fused = fuse_ifcnn(model, srcs)
    cv2.imwrite('output/ifcnn_test.png', fused)
    print('Test fused saved: output/ifcnn_test.png')
