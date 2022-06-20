tools/dist_train.sh \
configs/convnext/convnext_tt/upernet_convnext_tt_512_160k_ade20k_ms.py 8 \
--work-dir ./checkpoint/ \
--seed 0 \
--deterministic \
--options model.pretrained=/workdir/checkpoint/baseline/baseline_tiny0/checkpoint-best.pth