def init_datasets(args):
    size_hr = (args.resolution_hr, args.resolution_hr)
    size_lr = (args.resolution_lr, args.resolution_lr)
    
    # Matting datasets:
    if args.dataset == 'videomatte':
        dataset_lr_train = VideoMatteDataset(
            videomatte_dir=DATA_PATHS['videomatte']['train'],
            background_image_dir=DATA_PATHS['background_images']['train'],
            background_video_dir=DATA_PATHS['background_videos']['train'],
            size=args.resolution_lr,
            seq_length=args.seq_length_lr,
            seq_sampler=TrainFrameSampler(),
            transform=VideoMatteTrainAugmentation(size_lr))
        if args.train_hr:
            dataset_hr_train = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=args.resolution_hr,
                seq_length=args.seq_length_hr,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation(size_hr))
        dataset_valid = VideoMatteDataset(
            videomatte_dir=DATA_PATHS['videomatte']['valid'],
            background_image_dir=DATA_PATHS['background_images']['valid'],
            background_video_dir=DATA_PATHS['background_videos']['valid'],
            size=args.resolution_hr if args.train_hr else args.resolution_lr,
            seq_length=args.seq_length_hr if args.train_hr else args.seq_length_lr,
            seq_sampler=ValidFrameSampler(),
            transform=VideoMatteValidAugmentation(size_hr if args.train_hr else size_lr))
    else:
        dataset_lr_train = ImageMatteDataset(
            imagematte_dir=DATA_PATHS['imagematte']['train'],
            background_image_dir=DATA_PATHS['background_images']['train'],
            background_video_dir=DATA_PATHS['background_videos']['train'],
            size=args.resolution_lr,
            seq_length=args.seq_length_lr,
            seq_sampler=TrainFrameSampler(),
            transform=ImageMatteAugmentation(size_lr))
        if args.train_hr:
            dataset_hr_train = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['imagematte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=args.resolution_hr,
                seq_length=args.seq_length_hr,
                seq_sampler=TrainFrameSampler(),
                transform=ImageMatteAugmentation(size_hr))
        dataset_valid = ImageMatteDataset(
            imagematte_dir=DATA_PATHS['imagematte']['valid'],
            background_image_dir=DATA_PATHS['background_images']['valid'],
            background_video_dir=DATA_PATHS['background_videos']['valid'],
            size=args.resolution_hr if args.train_hr else args.resolution_lr,
            seq_length=args.seq_length_hr if args.train_hr else args.seq_length_lr,
            seq_sampler=ValidFrameSampler(),
            transform=ImageMatteAugmentation(size_hr if args.train_hr else size_lr))
        
    # Matting dataloaders:
    datasampler_lr_train = DistributedSampler(
        dataset=dataset_lr_train,
        rank=rank,
        num_replicas=world_size,
        shuffle=True)
    dataloader_lr_train = DataLoader(
        dataset=dataset_lr_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        sampler=datasampler_lr_train,
        pin_memory=True)
    if args.train_hr:
        datasampler_hr_train = DistributedSampler(
            dataset=dataset_hr_train,
            rank=rank,
            num_replicas=world_size,
            shuffle=True)
        dataloader_hr_train = DataLoader(
            dataset=dataset_hr_train,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            sampler=datasampler_hr_train,
            pin_memory=True)
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True)
    
    # Segementation datasets
    dataset_seg_image = ConcatDataset([
        CocoPanopticDataset(
            imgdir=DATA_PATHS['coco_panoptic']['imgdir'],
            anndir=DATA_PATHS['coco_panoptic']['anndir'],
            annfile=DATA_PATHS['coco_panoptic']['annfile'],
            transform=CocoPanopticTrainAugmentation(size_lr)),
        SuperviselyPersonDataset(
            imgdir=DATA_PATHS['spd']['imgdir'],
            segdir=DATA_PATHS['spd']['segdir'],
            transform=CocoPanopticTrainAugmentation(size_lr))
    ])
    datasampler_seg_image = DistributedSampler(
        dataset=dataset_seg_image,
        rank=rank,
        num_replicas=world_size,
        shuffle=True)
    dataloader_seg_image = DataLoader(
        dataset=dataset_seg_image,
        batch_size=args.batch_size_per_gpu * args.seq_length_lr,
        num_workers=args.num_workers,
        sampler=datasampler_seg_image,
        pin_memory=True)
    
    dataset_seg_video = YouTubeVISDataset(
        videodir=DATA_PATHS['youtubevis']['videodir'],
        annfile=DATA_PATHS['youtubevis']['annfile'],
        size=args.resolution_lr,
        seq_length=args.seq_length_lr,
        seq_sampler=TrainFrameSampler(speed=[1]),
        transform=YouTubeVISAugmentation(size_lr))
    datasampler_seg_video = DistributedSampler(
        dataset=dataset_seg_video,
        rank=rank,
        num_replicas=world_size,
        shuffle=True)
    dataloader_seg_video = DataLoader(
        dataset=dataset_seg_video,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        sampler=datasampler_seg_video,
        pin_memory=True)