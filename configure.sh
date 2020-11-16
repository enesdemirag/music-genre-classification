# Dependencies
pip3 install numpy matplotlib librosa tensorflow pandas pillow

# Dataset
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar -xzf genres.tar.gz
mkdir -p dataset/audios
mv genres/* -v dataset/audios/
rm -rf genres genres.tar.gz