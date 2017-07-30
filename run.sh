if [ $# -eq 2 ]; then
    # assign the provided arguments to variables
    do_training=$1
    input_filename=$2
else
    # assign the default values to variables
    do_training='train'
    development_dataset='data/development_sample_dataset_speaker.hdf5'
    enrollment_dataset='data/enrollment-evaluation_sample_dataset.hdf5'
    evaluation_dataset='data/enrollment-evaluation_sample_dataset.hdf5'
fi

if [ $do_training = 'train' ]; then

    # development
    python -u ./code/1-development/train_softmax.py --num_epochs=1 --batch_size=3 --development_dataset_path=$development_dataset --train_dir=results/TRAIN_CNN_3D/train_logs

    # enrollment - Automatically restore the latest checkpoint from all saved checkpoints
    python -u ./code/2-enrollment/enrollment.py --development_dataset_path=$development_dataset --enrollment_dataset_path=$enrollment_dataset --checkpoint_dir=results/TRAIN_CNN_3D/ --enrollment_dir=results/Model

    # evaluation
    python -u ./code/3-evaluation/evaluation.py --development_dataset_path=$development_dataset --evaluation_dataset_path=$evaluation_dataset --checkpoint_dir=results/TRAIN_CNN_3D/ --evaluation_dir=results/SCORES --enrollment_dir=results/Model

    # ROC curve
    python -u ./code/4-ROC_PR_curve/calculate_roc.py --evaluation_dir=results/SCORES

    # Plot ROC
    python -u ./code/4-ROC_PR_curve/PlotROC.py --evaluation_dir=results/SCORES --plot_dir=results/PLOTS

    # Plot ROC
    python -u ./code/4-ROC_PR_curve/PlotPR.py --evaluation_dir=results/SCORES --plot_dir=results/PLOTS

    # Plot HIST
    python -u ./code/4-ROC_PR_curve/PlotHIST.py --evaluation_dir=results/SCORES --plot_dir=results/PLOTS --num_bins=5



else

    echo "No training or testing will be performed!"

fi

