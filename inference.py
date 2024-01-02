import numpy as np
from models.detr import build
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import argparse


torch.set_grad_enabled(False)
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
transform_input = transforms.Compose([transforms.Resize(800),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device="cuda")
    return b


def plot_results(pil_img, prob, boxes, img_save_path):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=colors[cl], linewidth=3))
        
        text = f'{CLASSES[cl]}:      {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=9,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.savefig(img_save_path)
    plt.axis('off')
    plt.show()


def main(args):
    checkpoint_path, img_path, img_save_path=args.checkpoint_path,args.img_path,args.img_save_path
    a = torch.load(checkpoint_path)
    args = a['args']
    model = build(args)[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # 加载模型参数
    model_data = torch.load(checkpoint_path)['model']
    model.load_state_dict(model_data)

    model.eval()
    img = Image.open(img_path).convert('RGB')
    size = img.size
    
    inputs = transform_input(img).unsqueeze(0)
    outputs = model(inputs.to(device))
    # 这类最后[0, :, :-1]索引其实是把背景类筛选掉了
    probs = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # 可修改阈值,只输出概率大于0.7的物体
    keep = probs.max(-1).values > 0.7
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], size)
    # 保存输出结果
    ori_img = np.array(img)
    plot_results(ori_img, probs[keep], bboxes_scaled, img_save_path)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--checkpoint_path', default='output/detr_clean.pth', type=str)
    parser.add_argument('--img_path', default='test.jpg', type=str)
    parser.add_argument('--img_save_path', default='result2.jpg', type=str)
    return parser

# python inference.py 
if __name__ == "__main__":
    CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
    CLASSES = [
    'corn', 'weed']
    parser = argparse.ArgumentParser('DETR inference script', parents=[get_args_parser()])
    args = parser.parse_args()


    main(args)
    
