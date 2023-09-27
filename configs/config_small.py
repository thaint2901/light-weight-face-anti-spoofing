exp_num = 0

dataset = 'external'

multi_task_learning = True

evaluation = False

test_steps = None

datasets = dict(LCCFASD_root='./LCC_FASDcropped',
                Celeba_root='./CelebA_Spoof',
                Casia_root='./CASIA')

external = dict(train_params={
    "path_imgrec": "/mnt/nvme0n1p2/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/train_4.0.rec",
    "path_imgidx": "/mnt/nvme0n1p2/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/train_4.0.idx",
    "multi_learning": True,
    "scale": 1.5
},
                val_params=dict(), test_params=dict())

img_norm_cfg = dict(mean=[0.5295, 0.4319, 0.3978],
                    std=[0.2578, 0.2299, 0.2246])

optimizer = dict(lr=0.005, momentum=0.9, weight_decay=5e-4)

scheduler = dict(milestones=[20,50], gamma=0.2)

data = dict(batch_size=4,
            data_loader_workers=4,
            sampler=None,
            pin_memory=True)

resize = dict(height=224, width=224)

checkpoint = dict(snapshot_name="MobileNet3.pth.tar",
                  experiment_path='./logs')

loss = dict(loss_type='amsoftmax',
            amsoftmax=dict(m=0.5,
                           s=1,
                           margin_type='cross_entropy',
                           label_smooth=False,
                           smoothing=0.1,
                           ratio=[1,1],
                           gamma=0),
            soft_triple=dict(cN=2, K=10, s=1, tau=.2, m=0.35))

epochs = dict(start_epoch=0, max_epoch=71)

model= dict(model_type='Mobilenet3',
            model_size = 'small',
            width_mult = 1.0,
            pretrained=True,
            embeding_dim=1024,
            imagenet_weights='./pretrained/mobilenetv3-small-55df8e1f.pth')

aug = dict(type_aug=None,
            alpha=0.5,
            beta=0.5,
            aug_prob=0.7)

curves = dict(det_curve='det_curve_0.png',
              roc_curve='roc_curve_0.png')

dropout = dict(prob_dropout=0.1,
               classifier=0.1,
               type='bernoulli',
               mu=0.5,
               sigma=0.3)

data_parallel = dict(use_parallel=False,
                     parallel_params=dict(device_ids=[0], output_device=0))

RSC = dict(use_rsc=False,
           p=0.333,
           b=0.333)

test_dataset = dict(type='LCC_FASD')

conv_cd = dict(theta=0)
