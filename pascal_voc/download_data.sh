#echo "Creating data directory..."
#mkdir -p data && cd data
#mkdir pascal_voc

echo "Downloading Pascal VOC 2012 data..."
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar

echo "Extracting VOC data..."
tar xf VOCtrainval_06-Nov-2007.tar

echo "Remove the tat file"
rm VOCtrainval_06-Nov-2007.tar

echo "Done."
