

for WEIGHT in yolov8m_e1000s100_dp yolov8le1000s100_dp; do
    echo $WEIGHT
    cp weights/${WEIGHT}/weights/best.pt ${WEIGHT}.pt
    for EPOCH in 100 200 300 400 500 600 700 800 900; do
        cp weights/${WEIGHT}/weights/epoch${EPOCH}.pt ${WEIGHT}_epoch${EPOCH}.pt
    done
done