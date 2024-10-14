import paddle
def main(args):
    backbone = C3D(dropout_ratio=0.5, init_std=0.005)
    head = I3DHead(num_classes=101, in_channels=4096, spatial_type=None, dropout_ratio=0.5, init_std=0.01)
    net = Recognizer3D(backbone=backbone, cls_head=head)

    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        print("Loaded trained params of model successfully.")

    shape = [None, 1, 3, 16, 112, 112]
    new_net = net
    new_net.eval()
    # 将new_net转换为静态图
    new_net = paddle.jit.to_static(new_net, input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.save_dir, 'inference', "inference")
    paddle.jit.save(new_net, path=save_path)
    