# for batch_size in 128 256
# do
#     for threshold in 0.75 0.85 0.95
#     do
#         for mu in 1 3 7
#         do
#             python ssl_train.py \
#                 --epoch 100 \
#                 --batch_size $batch_size \
#                 --threshold $threshold \
#                 --mu $mu \
#                 --file_name ssl_b${batch_size}_t${threshold}_mu${mu}.tsv
#         done
#     done
# done

# for batch_size in 512
# do
#     for threshold in 0.75
#     do
#         for mu in 1
#         do
#             python ssl_train.py \
#                 --epoch 100 \
#                 --batch_size $batch_size \
#                 --threshold $threshold \
#                 --mu $mu \
#                 --file_name test.tsv
#         done
#     done
# done

# for batch_size in 512
# do
#     for threshold in 0.75
#     do
#         for mu in 3 7
#         do
#             python ssl_train.py \
#                 --epoch 100 \
#                 --batch_size $batch_size \
#                 --threshold $threshold \
#                 --mu $mu \
#                 --seed 1 \
#                 --file_name ssl_b${batch_size}_t${threshold}_mu${mu}_s1.tsv
#         done
#     done
# done

for batch_size in 512
do
    for threshold in 0.75 0.85 0.95
    do
        for seed in 0 1 2
        do
            python meps_train.py \
                --epoch 100 \
                --batch_size $batch_size \
                --threshold $threshold \
                --mu 3 \
                --seed ${seed} \
                --ema_momentum 0.999 \
                --temperature 0.5 \
                --similarity polynomial \
                --file_name meps_b${batch_size}_t${threshold}_mu3_te0.5_s${seed}.tsv
        done
    done
done