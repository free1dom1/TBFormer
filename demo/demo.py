from argparse import ArgumentParser
from mmseg.apis import inference_segmentor, init_segmentor
import matplotlib.pyplot as plt
import cv2


def main():
    parser = ArgumentParser()
    parser.add_argument('--imgname', default='demo/test-1.jpg', help='Test image name')
    parser.add_argument('--config', default='configs/tbformer_vit-b_8x1_512x512_300k_forgery.py', help='Config file')
    parser.add_argument('--checkpoint', default='checkpoint/TBFormer.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:1', help='Device used for inference')
    args = parser.parse_args()

    img = cv2.imread(args.imgname)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    result = inference_segmentor(model, args.imgname)
    pre_mask = result[0] * 255

    fig = plt.figure()
    plt.subplots_adjust(wspace=0.3)
    a = fig.add_subplot(1, 2, 1)
    a.set_title('input image')
    plt.imshow(img)
    b = fig.add_subplot(1, 2, 2)
    b.set_title('predicted mask')
    plt.imshow(pre_mask)
    plt.show()


if __name__ == '__main__':
    main()

