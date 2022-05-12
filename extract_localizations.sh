python3.7 test.py
mv /tmp/tmp* /home/balaji/manthan/tator/images;

for f in /home/balaji/manthan/tator/images/*; do
    mv "$f" "${f%}.jpg"
done;


