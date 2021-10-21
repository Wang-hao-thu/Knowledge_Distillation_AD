export PYTHONPATH=/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD:$PYTHONPATH
#imglist='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/data/jier/test/test.lst'
#imglist='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/Dataset/jier_data_5000/test_1.lst'
#imglist='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/data/jier/0928/test/test_v2.lst'
#imglist='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/Dataset/jier_data/test/test_270.lst'
imglist='/mnt/lustre/wanghao3/projects/DRAEM/data_path/ciwa_1/test/test.lst'
#imglist='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/data/jier/test_baoguo/test.lst'
#imglist='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/data/jier/test_baoguo/test_v2.lst'
#checkpoint='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/workdir/train_500/Cloner_0_epoch_400.pth'
#checkpoint='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/workdir/train_500/Cloner_0_epoch_600.pth'
#checkpoint='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/outputs/local_equal_net/ciwa_1/checkpoints/Cloner_0_epoch_280.pth'
checkpoint='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/outputs/local_equal_net/ciwa_1/checkpoints/Cloner_0_epoch_200.pth'
#checkpoint='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/outputs/local_equal_net/jier_data_5000/checkpoints/Cloner_0_epoch_10.pth'
config='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/configs/ciwa.yaml'

out_dir=$1
defect_miss=$out_dir/miss.txt
partition=mediaa
split_num=16

if [ -d $out_dir ]
then
    rm -rf $out_dir
fi
mkdir $out_dir -p
num=1
in_file=$imglist

out_file_dir=${out_dir}/tmp_dir
mkdir $out_file_dir
out_file=${out_file_dir}/imglist
total_line=$(wc -l < "$in_file")
lines=$(echo $total_line/$split_num | bc -l)
line=${lines%.*}
line=$(expr $line + $num)
echo $line
split -l $line -d $in_file $out_file
#######
result_dir=$out_dir/defect
vis_dir=$out_dir/vis
mkdir $result_dir -p
for each_file in `ls ${out_file_dir}`
do
{
    sub_img_list=${out_file_dir}/${each_file}
        srun -p $partition -x SH-IDC1-10-5-30-[69,102] -n1 --gres=gpu:1 \
        python /mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/infer.py \
        $result_dir/$each_file.txt \
        --config=$config \
        --imglist=$sub_img_list \
        --checkpoint=$checkpoint
} &
done
wait
echo done

result=$out_dir/results.all
cat $result_dir/* > $result
wc -l $result

python ./vis_result.py $result
