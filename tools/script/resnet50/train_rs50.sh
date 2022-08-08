tools/dist_train.sh \                                                                                                                               
configs/mobilenetv2/upernet_rs50_512_160k_ade20k_nms.py 8 \                                                                           
--work-dir ./checkpoint/upernet_rs50_512_160k_ade20k_nms  \                                                                                   
--seed 0 \                                                                                                                                                     
--deterministic \                                                                                                                                              
--options model.pretrained=/workdir/checkpoint/resnet50/rs50/checkpoint-best.pth 