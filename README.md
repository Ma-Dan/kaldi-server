# kaldi-server

Use [kaldi](https://github.com/kaldi-asr/kaldi) as ASR server with [DataTang Mandarin Model](http://www.kaldi-asr.org/models/m10).

使用[DataTang Mandarin Model 中文普通话](http://www.kaldi-asr.org/models/m10)的[Kaldi](https://github.com/kaldi-asr/kaldi) ASR服务器。

## Usage
1. Download Kaldi and modify code for server
* Assume running from path /kaldi-server
```shell
cd /
git clone https://github.com/Ma-Dan/kaldi-server
```

* Download kaldi and compile, assume kaldi in /kaldi path
```shell
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin golden
cd /kaldi/tools
apt install -y gcc make python unzip g++ automake autoconf sox gfortran libtool subversion zlib1g-dev
cd /data/kaldi/tools
make

cd ../tools; extras/install_openblas.sh

cd src/
./configure --shared --mathlib=OPENBLAS
make -j clean depend

# Modify online2-wav-nnet3-latgen-faster.cc to run a TCP server
cp /kaldi-server/kaldi/online2-wav-nnet3-latgen-faster.cc /kaldi/src/online2bin

make -j $(nproc)
```

2. Download DataTang Mandarin model

```shell
cd /kaldi/egs
wget http://www.kaldi-asr.org/models/10/0010_aidatatang_asr.tar.gz
tar xvf 0010_aidatatang_asr.tar.gz
```

3. Run kaldi server

Kaldi server will listen on 6666 port.

```shell
cd /kaldi-server/server
/kaldi/src/online2bin/online2-wav-nnet3-latgen-faster --do-endpointing=false --online=false --feature-type=mfcc --config=./online.conf --max-active=7000 --beam=15.0 --lattice-beam=6.0 --acoustic-scale=0.2 --word-symbol-table=/kaldi/egs/aidatatang_asr/exp/chain/tdnn_1a_sp/graph/words.txt /kaldi/egs/aidatatang_asr/exp/chain/tdnn_1a_sp/final.mdl /kaldi/egs/aidatatang_asr/exp/chain/tdnn_1a_sp/graph/HCLG.fst 'ark:echo utter1 utter1|' 'scp:echo utter1 null.wav|' ark:/dev/null
```

4. Run python server

```shell
cd /kaldi-server/server
python server.py --host_ip '0.0.0.0' --host_port 8000
```

5. Run python client and perform ASR

```shell
cd /kaldi-client
python client.py --host_ip '127.0.0.1' --host_port 8000
```

Press and release enter key to record a period of audio, client will send wav to server and return recognition result.
