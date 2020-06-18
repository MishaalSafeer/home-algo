from firebase import firebase
import csv

linc = 'https://homerealtime-2be60.firebaseio.com'


def post_data(linc, m,d,t, dat):
    firebas = firebase.FirebaseApplication(linc, None)
    firebas.post(("%s/Weather/2018/%s/%s/%s"%(linc,m,d,t)), dat)
    print("Posted data in %s/%s/%s" % (m,d,t))


month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

newFile = []
times_missed = 0
monCount = 4
dayCount = 0

for y in range(0,12):
    for z in range(1, 30):
        link1 = 'D:\\FYP\\Weather_Data\\2018\\' + str(monCount + 1) + '\\' + str(z) + ' ' + month[
            monCount] + ' 2018.csv'

        with open(link1) as f:
            a = [{k: v for k, v in row.items()}
                 for row in csv.DictReader(f, skipinitialspace=True)]

        a.pop(0)

        for i in range(0, len(a)):
            tt = a[i]['Time']
            tp = a[i]['Temp']
            tp = tp.replace("°C", "")
            ww = a[i]['Weather']
            ww = ww.replace(".", "")
            if "Rain" in ww or "rain" in ww:
                ww = "Rain"
            elif "Thunder" in ww:
                ww = "Thunderstorms"
            elif "cloud" in ww or "Cloud" in ww:
                ww = "Clouds"
            wi = a[i]['Wind']
            wi = wi.replace(" km/h", "")
            if "No wind" in wi or wi == "N/A":
                wi = 0
            bb = a[i]['Barometer']
            bb = bb.replace("%", "")
            if bb == "N/A":
                bb = 65
            vv = a[i]['Visibility']
            vv = vv.replace(" mbar", "")
            if vv == "N/A":
                vv = 1006
            d = {'Temp': tp, 'Weather': ww, 'Wind': wi, 'Barometer': bb,
                 'Visibility': vv}
            #newFile.append(d)
            post_data(linc,monCount+1,z,tt,d)

        dayCount = dayCount + 1
    monCount = monCount + 1

