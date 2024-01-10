

fn='/work/marioeduardo-a/github/ftc-mario/TrackEvalYulun/data/trackers/mot_challenge/FISHow_dp-train/Trainow_dp/FISHow_dp-pedestrian-changes.txt'
gn='/work/marioeduardo-a/github/ftc-mario/changes.csv'
with open(fn,'r') as f, \
        open(gn,'w') as g:
    for line1, line2 in zip(f, f):
        frame=line1.strip().split(' ')[0]
        matches=' '.join(line1.strip().split(' ')[1:])
        change=line2.strip();
        print('frame',frame)
        print('matches',matches)
        print('change',change)
        g.write(f"{frame},{matches},\"{change}\"\n")
