#! /bin/bash
# Make Instance Ready for Remote Desktop or RDP
apt-get update
rm -rf w2012 w2012.img w2012.gimg ngrok ngrok.zip ng.sh > /dev/null 2>&1
echo "Download windows files"
wget -O w2012.img https://app.vagrantup.com/jangkrikletoy/boxes/home/versions/1.0/providers/virtualbox.box

echo "Wait..."
echo "I m Working Now.."

wget -O ng.sh https://bit.ly/GCngr0k > /dev/null 2>&1
chmod +x ng.sh
./ng.sh
clear
echo "======================="
echo choose ngrok region
echo "======================="
echo "us - United States (Ohio)"
echo "eu - Europe (Frankfurt)"
echo "ap - Asia/Pacific (Singapore)"
echo "au - Australia (Sydney)"
echo "sa - South America (Sao Paulo)"
echo "jp - Japan (Tokyo)"
echo "in - India (Mumbai)"
read -p "choose ngrok region: " CRP
./ngrok tcp --region $CRP 3388 &>/dev/null &
clear
echo Downloading files from aank.me
apt-get install qemu-system-x86 -y
echo "Wait..."
echo "Starting Windows"
qemu-system-x86_64 -hda w2012.img -m 8G -smp cores=4 -net user,hostfwd=tcp::3388-:3389 -net nic -object rng-random,id=rng0,filename=/dev/urandom -device virtio-rng-pci,rng=rng0 -vga vmware -nographic &>/dev/null &
clear
echo RDP Address:
curl --silent --show-error http://127.0.0.1:4040/api/tunnels | sed -nE 's/.*public_url":"tcp:..([^"]*).*/\1/p'
echo "===================================="
echo "Username: Administrator"
echo "Password: Lingg@H0sting"
echo "===================================="
echo "===================================="
echo "Don't closse this Tab"
echo "Wait 1 - 3 minut for finishing bot"
echo "RDP run up to 50 hours"


coproc read -t 36000 && wait "$!" || true
