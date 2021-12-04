for batch_size in 512
do
    for threshold in 0.75
    do
        for seed in 0 1 2
        do
            for ulambda in 0.5
            do
                python ssl_train.py \
                    --epoch 100 \
                    --batch_size $batch_size \
                    --threshold $threshold \
                    --mu 3 \
                    --seed $seed \
                    --ema_momentum 0.999 \
                    --ulambda $ulambda \
                    --file_name ssl_b${batch_size}_t${threshold}_mu3_l${ulambda}.tsv
            done
        done
    done
done


# for batch_size in 512
# do
#     for threshold in 0.45 0.65
#     do
#         for seed in 0 1 2
#         do
#             python meps_train.py \
#                 --epoch 100 \
#                 --batch_size $batch_size \
#                 --threshold $threshold \
#                 --mu 3 \
#                 --seed ${seed} \
#                 --ema_momentum 0.999 \
#                 --temperature 0.05 \
#                 --file_name meps_b${batch_size}_t${threshold}_mu3_t0.05_s${seed}.tsv
#         done
#     done
# done