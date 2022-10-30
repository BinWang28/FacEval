
echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
start=`date +%s`
echo "= = = = = = = = = = = = = ="



python fac_gen_src/main.py \
    --neg_marker _neg \
    --output_dir ./scores_log/ \
    --load_file ./data/faceval_samples.json \
    --text_column dialogue \
    --summary_column summary \
    --model_name_or_path ./trained_model/ \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --length_penalty 1.0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345


python fac_gen_src/main.py \
    --neg_marker ss_neg \
    --output_dir ./scores_log/ \
    --load_file ./data/faceval_samples.json \
    --text_column dialogue \
    --summary_column summary \
    --model_name_or_path ./trained_model/ \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --length_penalty 1.0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345

python fac_gen_src/main.py \
    --neg_marker es_neg \
    --output_dir ./scores_log/ \
    --load_file ./data/faceval_samples.json \
    --text_column dialogue \
    --summary_column summary \
    --model_name_or_path ./trained_model/ \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --length_penalty 1.0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345


python fac_gen_src/main.py \
    --neg_marker ps_neg \
    --output_dir ./scores_log/ \
    --load_file ./data/faceval_samples.json \
    --text_column dialogue \
    --summary_column summary \
    --model_name_or_path ./trained_model/ \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --length_penalty 1.0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345



python fac_gen_src/main.py \
    --neg_marker ds_neg \
    --output_dir ./scores_log/ \
    --load_file ./data/faceval_samples.json \
    --text_column dialogue \
    --summary_column summary \
    --model_name_or_path ./trained_model/ \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --length_penalty 1.0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345



python fac_gen_src/main.py \
    --neg_marker ns_neg \
    --output_dir ./scores_log/ \
    --load_file ./data/faceval_samples.json \
    --text_column dialogue \
    --summary_column summary \
    --model_name_or_path ./trained_model/ \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --length_penalty 1.0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345


python fac_gen_src/main.py \
    --neg_marker ng_neg \
    --output_dir ./scores_log/ \
    --load_file ./data/faceval_samples.json \
    --text_column dialogue \
    --summary_column summary \
    --model_name_or_path ./trained_model/ \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --length_penalty 1.0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345


echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="