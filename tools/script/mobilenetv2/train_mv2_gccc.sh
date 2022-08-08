tools/dist_train.sh \                                                                                                                               
configs/mobilenetv2/upernet_mv2_gcc_512_160k_ade20k_nms.py 8 \                                                                           
--work-dir ./checkpoint/upernet_mv2_gcc_512_160k_ade20k_nms  \                                                                                   
--seed 0 \                                                                                                                                                     
--deterministic \                                                                                                                                              
--options model.pretrained=/workdir/checkpoint/mobilenetv2/mv2_gcc/checkpoint-best_mmdet.pth 