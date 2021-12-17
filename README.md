# a

#-----------------------背景色や照明の当て方などの撮影条件を判定する関数-----------------------

#OpenCVのread関数でフレーム配列を読み出しshape関数で横幅を取得。
#取得した横幅と両サイド何％で判定したいかを引数で指定する。
#返り値である差分の確率(probabilityOfDifference)は検証しながらトリガラインを決めよう！！
def checkTheBackgroundColor(frame,ratio):
    
        #サイズ取得
        height, width, channels = frame.shape[:3]
    
        #切り出し範囲を計算
        cutoutWidthRange = (width * (ratio / 100)) / 2
​
        #差分を絶対値で取得。差が無いほど0に近く、背景色が統一されていると判断できる。
        arrayDifference = np.abs(frame[0 : height, 0:int(cutoutWidthRange) ]-frame[0 : height, int(width-cutoutWidthRange):width ])
        
        #確率に変換する。0に近いほど差がない。
        probabilityOfDifference = ((np.count_nonzero(arrayDifference == 0))/arrayDifference.size)*100
        
        return  probabilityOfDifference
#-----------------------マノメータを中心に位置して撮影できているか判定する関数-----------------------

        rightArrayDifference = np.abs(frame[0 : height, int(center) - int(cutoutWidthRange / 2) : int(center) + int(cutoutWidthRange / 2)]-frame[0 : height, 0:int(cutoutWidthRange) ])
        leftArrayDifference = np.abs(frame[0 : height, int(center) - int(cutoutWidthRange / 2) : int(center) + int(cutoutWidthRange / 2)]-frame[0 : height, int(width-cutoutWidthRange):width ])
  
#OpenCVのread関数でフレーム配列を読み出しshape関数で横幅を取得。
#取得した横幅と中心何％で判定したいかを引数で指定する。
#返り値である差分の確率(rightProbabilityOfDifference,leftProbabilityOfDifference)は検証しながらトリガラインを決めよう！！
#rightProbabilityOfDifferenceが高いと右にズレている。eftProbabilityOfDifferenceが高いと左にズレている。
def checkThePositionOfTheManometer(frame,ratio):
        
        #サイズ取得
        height, width, channels = frame.shape[:3]
        center = width / 2
        
        #切り出し範囲を計算
        cutoutWidthRange = (width * (ratio / 100)) / 2
        
        #左右の差分を絶対値で取得。差が無いほど0に近く、背景色が統一されていると判断できる。
        rightArrayDifference = np.abs(frame[0 : height, int(center) - int(cutoutWidthRange / 2) : int(center) + int(cutoutWidthRange / 2)]-frame[0 : height, 0:int(cutoutWidthRange) ])
        leftArrayDifference = np.abs(frame[0 : height, int(center) - int(cutoutWidthRange / 2) : int(center) + int(cutoutWidthRange / 2)]-frame[0 : height, int(width-cutoutWidthRange):width ])
    
        #確率に変換する。0に近いほど差がない。
        rightProbabilityOfDifference = ((np.count_nonzero(rightArrayDifference == 0))/rightArrayDifference.size)*100
        leftProbabilityOfDifference = ((np.count_nonzero(leftArrayDifference == 0))/leftArrayDifference.size)*100
        
        return  rightProbabilityOfDifference,leftProbabilityOfDifference
#-----------------------撮影時の注意点を表示する関数-----------------------

def precautionsWhenShooting(frame,flagSideDiff,flagCenterDiff,ratio):
    
        #サイズ取得
        height, width, channels = frame.shape[:3]
        center = width / 2
        
        if flagSideDiff == 0: #左右の背景色がOK
            
                if flagCenterDiff == 0: #センター位置もOK
                
                    frame = frame
                    confirmationOfAllConditionsFlag = 0
                    
                else: #左右にズレている場合
                     
                    #ズレ感知の範囲
                    frame = cv2.line(frame,(int(center-(width * (ratio / 100)) / 2),0),(int(center-(width * (ratio / 100)) / 2),height),(0,0, 255),1)
                    frame = cv2.line(frame,(int(center+(width * (ratio / 100)) / 2),0),(int(center+(width * (ratio / 100)) / 2),height),(0,0, 255),1)
​
                    #注意コメント
                    cv2.putText(frame,'<< Put the',(330, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 5, cv2.LINE_AA)
                    cv2.putText(frame,'   background',(340, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 5, cv2.LINE_AA)
                    cv2.putText(frame,'   in the area',(340, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 5, cv2.LINE_AA)
                    confirmationOfAllConditionsFlag = 1
​
        else: #左右の背景色がNG
        
            #ズレ感知の範囲
            frame = cv2.line(frame,(int((width * (ratio / 100)) / 2),0),(int((width * (ratio / 100)) / 2),height),(0,0, 255),1)
            frame = cv2.line(frame,(int(width-(width * (ratio / 100)) / 2),0),(int(width-(width * (ratio / 100)) / 2),height),(0,0, 255),1)
            
            #注意コメント
            cv2.putText(frame,'<< Side background color is NG >>',(30, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 5, cv2.LINE_AA)
            confirmationOfAllConditionsFlag = 1
            
​
        return frame,confirmationOfAllConditionsFlag
#-----------------------スケールを感知する関数-----------------------

#-----------------------ライブラリのインポート-----------------------
#画像の分割に必要
import cv2
from datetime import datetime
​
#画像の分割や判定に必要
import numpy as np
from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )
import matplotlib.pyplot as plt
# イメージ画像のロードするための関数
from tensorflow.keras.preprocessing.image import load_img
# イメージ画像の配列処理するための関数
from tensorflow.keras.preprocessing.image import img_to_array
# 予測結果を復元するための関数
from tensorflow.keras.applications.imagenet_utils import decode_predictions
​
#-----------------------モデルをダウンロード-----------------------
vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')
​
#-----------------------画像を判定する-----------------------
def rulerJudgment(frame):
​
    # 画像を判定用に処理
    resFre = cv2.resize(frame,dsize=(224, 224)) #リサイズ
    resFre = resFre.reshape(-1,224,224,3) #model.predictで必要な形式は4次元のため3次元を4次元に変換
    #resFre = resFre.astype('float32')/255.0 #正規化不要
​
    # VGG16のモデルと比較・判定する
    processed_image = vgg16.preprocess_input(resFre.copy())
    predictions = vgg_model.predict(processed_image)
    label_vgg = decode_predictions(predictions)
​
    #結果を読み込み
    for prediction_id in range(len(label_vgg[0])):
    
        #確率の高い順5位まで表示
        #print(label_vgg[0][prediction_id])
    
        break
​
    #1位の結果を表示
    x = str(label_vgg[0][prediction_id])
​
    #一番高い確率のカテゴリを表示
    return x
#-----------------------スケールを塗りつぶす関数-----------------------

def fillTheRuler(frame,confirmationOfAllConditionsFlag):
    
        height, width, channels = frame.shape[:3]
        center = width / 2
        
        fillImg = frame[0 : height, int(center-1): int(center+1)] #センターラインの配列を取得
        
        AveBGR =np.mean(fillImg, axis=(0,1)) #センターラインのBGR(青緑赤)の平均値を取得　
​
        pts = np.array( [ [int(center),0], [int(center),height], [width, height], [width,0] ] )#塗りつぶしのエリアを取得
        
        #右端エリアを判定
        x = rulerJudgment(frame);
        
        #結果表示
        cv2.putText(frame,str(x[14:25]),(30, 300), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5, cv2.LINE_AA)
        
        frame = cv2.fillPoly(frame, pts =[pts], color=(AveBGR)) #センターラインより右を塗りつぶし
        
        return frame
#-----------------------撮影時の関数-----------------------

def cameraMovie(camera_num,backgroundColorRatio):
​
    cap = cv2.VideoCapture(camera_num)
    
    while(True):
    
        ret, frame = cap.read()
        
        #-----------------------背景色や照明の当て方をチェック-----------------------
        probabilityOfDifference = checkTheBackgroundColor(frame,backgroundColorRatio);
        
        if probabilityOfDifference < 1:
        
            flagSideDiff = 0 #OK
            
        else:
            flagSideDiff = 1 #NG
            
        #-----------------------マノメータの中心位置を確認-----------------------
        rightProbabilityOfDifference,leftProbabilityOfDifference = checkThePositionOfTheManometer(frame,backgroundColorRatio);
​
        #出力値を確認する
        #print(rightProbabilityOfDifference,leftProbabilityOfDifference)
        
        if rightProbabilityOfDifference != 0 and leftProbabilityOfDifference != 0:
            
            flagCenterDiff = 0 #OK
            
        else:
            
            flagCenterDiff = 1 #ズレている
​
        #-----------------------撮影時の注意点を表示する-----------------------
        frame , confirmationOfAllConditionsFlag = precautionsWhenShooting(frame,flagSideDiff,flagCenterDiff,backgroundColorRatio);
​
        if confirmationOfAllConditionsFlag == 0:
            
            #-----------------------スケールを塗りつぶす-----------------------
            frame =  fillTheRuler(frame,confirmationOfAllConditionsFlag);
​
            cv2.imshow('frame show(shot:key"s"_exit:key"e")',frame)
            
        else:
            cv2.imshow('frame show(shot:key"s"_exit:key"e")',frame)
        
        key = cv2.waitKey(1) & 0xFF #0xFFはキーボード入力を正しく読み取るため末尾から読む
    
    
        # 撮影ボタンが押されたら
        if key == ord('s'):
​
            break
        elif key == ord('e'):
            cv2.destroyAllWindows()  
            break
import cv2
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
import numpy as np
def main():
    
    camera_num = 0# 外付けカメラは1
    
    backgroundColorRatio = 5# 画像両サイドから背景色の統一性を判断するため割合を指定
    
    # 上記の情報を撮影関数に反映
    cameraMovie(camera_num,backgroundColorRatio)

if __name__ == '__main__':
    main()
def main():
    
    camera_num = 0# 外付けカメラは1
    
    backgroundColorRatio = 5# 画像両サイドから背景色の統一性を判断するため割合を指定
    
    # 上記の情報を撮影関数に反映
    cameraMovie(camera_num,backgroundColorRatio)
​
if __name__ == '__main__':
    main()

#-----------------------カテゴリのリスト-----------------------

1000 synsets for Task 2 (same as in ILSVRC2012)
	n02119789	:	0	kit fox, Vulpes macrotis
	n02100735	:	1	English setter
	n02096294	:	2	Australian terrier
	n02066245	:	3	grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus
	n02509815	:	4	lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens
	n02124075	:	5	Egyptian cat
	n02417914	:	6	ibex, Capra ibex
	n02123394	:	7	Persian cat
	n02125311	:	8	cougar, puma, catamount, mountain lion, painter, panther, Felis concolor
	n02423022	:	9	gazelle
	n02346627	:	10	porcupine, hedgehog
	n02077923	:	11	sea lion
	n02447366	:	12	badger
	n02109047	:	13	Great Dane
	n02092002	:	14	Scottish deerhound, deerhound
	n02071294	:	15	killer whale, killer, orca, grampus, sea wolf, Orcinus orca
	n02442845	:	16	mink
	n02504458	:	17	African elephant, Loxodonta africana
	n02114712	:	18	red wolf, maned wolf, Canis rufus, Canis niger
	n02128925	:	19	jaguar, panther, Panthera onca, Felis onca
	n02117135	:	20	hyena, hyaena
	n02493509	:	21	titi, titi monkey
	n02457408	:	22	three-toed sloth, ai, Bradypus tridactylus
	n02389026	:	23	sorrel
	n02443484	:	24	black-footed ferret, ferret, Mustela nigripes
	n02110341	:	25	dalmatian, coach dog, carriage dog
	n02093256	:	26	Staffordshire bullterrier, Staffordshire bull terrier
	n02106382	:	27	Bouvier des Flandres, Bouviers des Flandres
	n02441942	:	28	weasel
	n02113712	:	29	miniature poodle
	n02415577	:	30	bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis
	n02356798	:	31	fox squirrel, eastern fox squirrel, Sciurus niger
	n02488702	:	32	colobus, colobus monkey
	n02123159	:	33	tiger cat
	n02422699	:	34	impala, Aepyceros melampus
	n02114855	:	35	coyote, prairie wolf, brush wolf, Canis latrans
	n02094433	:	36	Yorkshire terrier
	n02111277	:	37	Newfoundland, Newfoundland dog
	n02119022	:	38	red fox, Vulpes vulpes
	n02422106	:	39	hartebeest
	n02120505	:	40	grey fox, gray fox, Urocyon cinereoargenteus
	n02086079	:	41	Pekinese, Pekingese, Peke
	n02484975	:	42	guenon, guenon monkey
	n02137549	:	43	mongoose
	n02500267	:	44	indri, indris, Indri indri, Indri brevicaudatus
	n02129604	:	45	tiger, Panthera tigris
	n02396427	:	46	wild boar, boar, Sus scrofa
	n02391049	:	47	zebra
	n02412080	:	48	ram, tup
	n02480495	:	49	orangutan, orang, orangutang, Pongo pygmaeus
	n02110806	:	50	basenji
	n02128385	:	51	leopard, Panthera pardus
	n02100583	:	52	vizsla, Hungarian pointer
	n02494079	:	53	squirrel monkey, Saimiri sciureus
	n02123597	:	54	Siamese cat, Siamese
	n02481823	:	55	chimpanzee, chimp, Pan troglodytes
	n02105505	:	56	komondor
	n02489166	:	57	proboscis monkey, Nasalis larvatus
	n02364673	:	58	guinea pig, Cavia cobaya
	n02114548	:	59	white wolf, Arctic wolf, Canis lupus tundrarum
	n02134084	:	60	ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus
	n02480855	:	61	gorilla, Gorilla gorilla
	n02403003	:	62	ox
	n02108551	:	63	Tibetan mastiff
	n02493793	:	64	spider monkey, Ateles geoffroyi
	n02107142	:	65	Doberman, Doberman pinscher
	n02397096	:	66	warthog
	n02437312	:	67	Arabian camel, dromedary, Camelus dromedarius
	n02483708	:	68	siamang, Hylobates syndactylus, Symphalangus syndactylus
	n02099601	:	69	golden retriever
	n02106166	:	70	Border collie
	n02326432	:	71	hare
	n02108089	:	72	boxer
	n02486261	:	73	patas, hussar monkey, Erythrocebus patas
	n02486410	:	74	baboon
	n02487347	:	75	macaque
	n02492035	:	76	capuchin, ringtail, Cebus capucinus
	n02099267	:	77	flat-coated retriever
	n02395406	:	78	hog, pig, grunter, squealer, Sus scrofa
	n02109961	:	79	Eskimo dog, husky
	n02101388	:	80	Brittany spaniel
	n03187595	:	81	dial telephone, dial phone
	n03733281	:	82	maze, labyrinth
	n02101006	:	83	Gordon setter
	n02115641	:	84	dingo, warrigal, warragal, Canis dingo
	n02342885	:	85	hamster
	n02120079	:	86	Arctic fox, white fox, Alopex lagopus
	n02408429	:	87	water buffalo, water ox, Asiatic buffalo, Bubalus bubalis
	n02133161	:	88	American black bear, black bear, Ursus americanus, Euarctos americanus
	n02328150	:	89	Angora, Angora rabbit
	n02410509	:	90	bison
	n02492660	:	91	howler monkey, howler
	n02398521	:	92	hippopotamus, hippo, river horse, Hippopotamus amphibius
	n02510455	:	93	giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca
	n02123045	:	94	tabby, tabby cat
	n02490219	:	95	marmoset
	n02109525	:	96	Saint Bernard, St Bernard
	n02454379	:	97	armadillo
	n02090379	:	98	redbone
	n02443114	:	99	polecat, fitch, foulmart, foumart, Mustela putorius
	n02361337	:	100	marmot
	n02483362	:	101	gibbon, Hylobates lar
	n02437616	:	102	llama
	n02325366	:	103	wood rabbit, cottontail, cottontail rabbit
	n02129165	:	104	lion, king of beasts, Panthera leo
	n02100877	:	105	Irish setter, red setter
	n02074367	:	106	dugong, Dugong dugon
	n02504013	:	107	Indian elephant, Elephas maximus
	n02363005	:	108	beaver
	n02497673	:	109	Madagascar cat, ring-tailed lemur, Lemur catta
	n02087394	:	110	Rhodesian ridgeback
	n02127052	:	111	lynx, catamount
	n02116738	:	112	African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus
	n02488291	:	113	langur
	n02114367	:	114	timber wolf, grey wolf, gray wolf, Canis lupus
	n02130308	:	115	cheetah, chetah, Acinonyx jubatus
	n02134418	:	116	sloth bear, Melursus ursinus, Ursus ursinus
	n02106662	:	117	German shepherd, German shepherd dog, German police dog, alsatian
	n02444819	:	118	otter
	n01882714	:	119	koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus
	n01871265	:	120	tusker
	n01872401	:	121	echidna, spiny anteater, anteater
	n01877812	:	122	wallaby, brush kangaroo
	n01873310	:	123	platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus
	n01883070	:	124	wombat
	n04086273	:	125	revolver, six-gun, six-shooter
	n04507155	:	126	umbrella
	n04147183	:	127	schooner
	n04254680	:	128	soccer ball
	n02672831	:	129	accordion, piano accordion, squeeze box
	n02219486	:	130	ant, emmet, pismire
	n02317335	:	131	starfish, sea star
	n01968897	:	132	chambered nautilus, pearly nautilus, nautilus
	n03452741	:	133	grand piano, grand
	n03642806	:	134	laptop, laptop computer
	n07745940	:	135	strawberry
	n02690373	:	136	airliner
	n04552348	:	137	warplane, military plane
	n02692877	:	138	airship, dirigible
	n02782093	:	139	balloon
	n04266014	:	140	space shuttle
	n03344393	:	141	fireboat
	n03447447	:	142	gondola
	n04273569	:	143	speedboat
	n03662601	:	144	lifeboat
	n02951358	:	145	canoe
	n04612504	:	146	yawl
	n02981792	:	147	catamaran
	n04483307	:	148	trimaran
	n03095699	:	149	container ship, containership, container vessel
	n03673027	:	150	liner, ocean liner
	n03947888	:	151	pirate, pirate ship
	n02687172	:	152	aircraft carrier, carrier, flattop, attack aircraft carrier
	n04347754	:	153	submarine, pigboat, sub, U-boat
	n04606251	:	154	wreck
	n03478589	:	155	half track
	n04389033	:	156	tank, army tank, armored combat vehicle, armoured combat vehicle
	n03773504	:	157	missile
	n02860847	:	158	bobsled, bobsleigh, bob
	n03218198	:	159	dogsled, dog sled, dog sleigh
	n02835271	:	160	bicycle-built-for-two, tandem bicycle, tandem
	n03792782	:	161	mountain bike, all-terrain bike, off-roader
	n03393912	:	162	freight car
	n03895866	:	163	passenger car, coach, carriage
	n02797295	:	164	barrow, garden cart, lawn cart, wheelbarrow
	n04204347	:	165	shopping cart
	n03791053	:	166	motor scooter, scooter
	n03384352	:	167	forklift
	n03272562	:	168	electric locomotive
	n04310018	:	169	steam locomotive
	n02704792	:	170	amphibian, amphibious vehicle
	n02701002	:	171	ambulance
	n02814533	:	172	beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
	n02930766	:	173	cab, hack, taxi, taxicab
	n03100240	:	174	convertible
	n03594945	:	175	jeep, landrover
	n03670208	:	176	limousine, limo
	n03770679	:	177	minivan
	n03777568	:	178	Model T
	n04037443	:	179	racer, race car, racing car
	n04285008	:	180	sports car, sport car
	n03444034	:	181	go-kart
	n03445924	:	182	golfcart, golf cart
	n03785016	:	183	moped
	n04252225	:	184	snowplow, snowplough
	n03345487	:	185	fire engine, fire truck
	n03417042	:	186	garbage truck, dustcart
	n03930630	:	187	pickup, pickup truck
	n04461696	:	188	tow truck, tow car, wrecker
	n04467665	:	189	trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi
	n03796401	:	190	moving van
	n03977966	:	191	police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria
	n04065272	:	192	recreational vehicle, RV, R.V.
	n04335435	:	193	streetcar, tram, tramcar, trolley, trolley car
	n04252077	:	194	snowmobile
	n04465501	:	195	tractor
	n03776460	:	196	mobile home, manufactured home
	n04482393	:	197	tricycle, trike, velocipede
	n04509417	:	198	unicycle, monocycle
	n03538406	:	199	horse cart, horse-cart
	n03788365	:	200	mosquito net
	n03868242	:	201	oxcart
	n02804414	:	202	bassinet
	n03125729	:	203	cradle
	n03131574	:	204	crib, cot
	n03388549	:	205	four-poster
	n02870880	:	206	bookcase
	n03018349	:	207	china cabinet, china closet
	n03742115	:	208	medicine chest, medicine cabinet
	n03016953	:	209	chiffonier, commode
	n04380533	:	210	table lamp
	n03337140	:	211	file, file cabinet, filing cabinet
	n03902125	:	212	pay-phone, pay-station
	n03891251	:	213	park bench
	n02791124	:	214	barber chair
	n04429376	:	215	throne
	n03376595	:	216	folding chair
	n04099969	:	217	rocking chair, rocker
	n04344873	:	218	studio couch, day bed
	n04447861	:	219	toilet seat
	n03179701	:	220	desk
	n03982430	:	221	pool table, billiard table, snooker table
	n03201208	:	222	dining table, board
	n03290653	:	223	entertainment center
	n04550184	:	224	wardrobe, closet, press
	n07742313	:	225	Granny Smith
	n07747607	:	226	orange
	n07749582	:	227	lemon
	n07753113	:	228	fig
	n07753275	:	229	pineapple, ananas
	n07753592	:	230	banana
	n07754684	:	231	jackfruit, jak, jack
	n07760859	:	232	custard apple
	n07768694	:	233	pomegranate
	n12267677	:	234	acorn
	n12620546	:	235	hip, rose hip, rosehip
	n13133613	:	236	ear, spike, capitulum
	n11879895	:	237	rapeseed
	n12144580	:	238	corn
	n12768682	:	239	buckeye, horse chestnut, conker
	n03854065	:	240	organ, pipe organ
	n04515003	:	241	upright, upright piano
	n03017168	:	242	chime, bell, gong
	n03249569	:	243	drum, membranophone, tympan
	n03447721	:	244	gong, tam-tam
	n03720891	:	245	maraca
	n03721384	:	246	marimba, xylophone
	n04311174	:	247	steel drum
	n02787622	:	248	banjo
	n02992211	:	249	cello, violoncello
	n03637318	:	250	lampshade, lamp shade
	n03495258	:	251	harp
	n02676566	:	252	acoustic guitar
	n03272010	:	253	electric guitar
	n03110669	:	254	cornet, horn, trumpet, trump
	n03394916	:	255	French horn, horn
	n04487394	:	256	trombone
	n03494278	:	257	harmonica, mouth organ, harp, mouth harp
	n03840681	:	258	ocarina, sweet potato
	n03884397	:	259	panpipe, pandean pipe, syrinx
	n02804610	:	260	bassoon
	n04141076	:	261	sax, saxophone
	n03372029	:	262	flute, transverse flute
	n11939491	:	263	daisy
	n12057211	:	264	yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum
	n09246464	:	265	cliff, drop, drop-off
	n09468604	:	266	valley, vale
	n09193705	:	267	alp
	n09472597	:	268	volcano
	n09399592	:	269	promontory, headland, head, foreland
	n09421951	:	270	sandbar, sand bar
	n09256479	:	271	coral reef
	n09332890	:	272	lakeside, lakeshore
	n09428293	:	273	seashore, coast, seacoast, sea-coast
	n09288635	:	274	geyser
	n03498962	:	275	hatchet
	n03041632	:	276	cleaver, meat cleaver, chopper
	n03658185	:	277	letter opener, paper knife, paperknife
	n03954731	:	278	plane, carpenter's plane, woodworking plane
	n03995372	:	279	power drill
	n03649909	:	280	lawn mower, mower
	n03481172	:	281	hammer
	n03109150	:	282	corkscrew, bottle screw
	n02951585	:	283	can opener, tin opener
	n03970156	:	284	plunger, plumber's helper
	n04154565	:	285	screwdriver
	n04208210	:	286	shovel
	n03967562	:	287	plow, plough
	n03000684	:	288	chain saw, chainsaw
	n01514668	:	289	cock
	n01514859	:	290	hen
	n01518878	:	291	ostrich, Struthio camelus
	n01530575	:	292	brambling, Fringilla montifringilla
	n01531178	:	293	goldfinch, Carduelis carduelis
	n01532829	:	294	house finch, linnet, Carpodacus mexicanus
	n01534433	:	295	junco, snowbird
	n01537544	:	296	indigo bunting, indigo finch, indigo bird, Passerina cyanea
	n01558993	:	297	robin, American robin, Turdus migratorius
	n01560419	:	298	bulbul
	n01580077	:	299	jay
	n01582220	:	300	magpie
	n01592084	:	301	chickadee
	n01601694	:	302	water ouzel, dipper
	n01608432	:	303	kite
	n01614925	:	304	bald eagle, American eagle, Haliaeetus leucocephalus
	n01616318	:	305	vulture
	n01622779	:	306	great grey owl, great gray owl, Strix nebulosa
	n01795545	:	307	black grouse
	n01796340	:	308	ptarmigan
	n01797886	:	309	ruffed grouse, partridge, Bonasa umbellus
	n01798484	:	310	prairie chicken, prairie grouse, prairie fowl
	n01806143	:	311	peacock
	n01806567	:	312	quail
	n01807496	:	313	partridge
	n01817953	:	314	African grey, African gray, Psittacus erithacus
	n01818515	:	315	macaw
	n01819313	:	316	sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
	n01820546	:	317	lorikeet
	n01824575	:	318	coucal
	n01828970	:	319	bee eater
	n01829413	:	320	hornbill
	n01833805	:	321	hummingbird
	n01843065	:	322	jacamar
	n01843383	:	323	toucan
	n01847000	:	324	drake
	n01855032	:	325	red-breasted merganser, Mergus serrator
	n01855672	:	326	goose
	n01860187	:	327	black swan, Cygnus atratus
	n02002556	:	328	white stork, Ciconia ciconia
	n02002724	:	329	black stork, Ciconia nigra
	n02006656	:	330	spoonbill
	n02007558	:	331	flamingo
	n02009912	:	332	American egret, great white heron, Egretta albus
	n02009229	:	333	little blue heron, Egretta caerulea
	n02011460	:	334	bittern
	n02012849	:	335	crane
	n02013706	:	336	limpkin, Aramus pictus
	n02018207	:	337	American coot, marsh hen, mud hen, water hen, Fulica americana
	n02018795	:	338	bustard
	n02025239	:	339	ruddy turnstone, Arenaria interpres
	n02027492	:	340	red-backed sandpiper, dunlin, Erolia alpina
	n02028035	:	341	redshank, Tringa totanus
	n02033041	:	342	dowitcher
	n02037110	:	343	oystercatcher, oyster catcher
	n02017213	:	344	European gallinule, Porphyrio porphyrio
	n02051845	:	345	pelican
	n02056570	:	346	king penguin, Aptenodytes patagonica
	n02058221	:	347	albatross, mollymawk
	n01484850	:	348	great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
	n01491361	:	349	tiger shark, Galeocerdo cuvieri
	n01494475	:	350	hammerhead, hammerhead shark
	n01496331	:	351	electric ray, crampfish, numbfish, torpedo
	n01498041	:	352	stingray
	n02514041	:	353	barracouta, snoek
	n02536864	:	354	coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch
	n01440764	:	355	tench, Tinca tinca
	n01443537	:	356	goldfish, Carassius auratus
	n02526121	:	357	eel
	n02606052	:	358	rock beauty, Holocanthus tricolor
	n02607072	:	359	anemone fish
	n02643566	:	360	lionfish
	n02655020	:	361	puffer, pufferfish, blowfish, globefish
	n02640242	:	362	sturgeon
	n02641379	:	363	gar, garfish, garpike, billfish, Lepisosteus osseus
	n01664065	:	364	loggerhead, loggerhead turtle, Caretta caretta
	n01667114	:	365	mud turtle
	n01667778	:	366	terrapin
	n01669191	:	367	box turtle, box tortoise
	n01675722	:	368	banded gecko
	n01677366	:	369	common iguana, iguana, Iguana iguana
	n01682714	:	370	American chameleon, anole, Anolis carolinensis
	n01685808	:	371	whiptail, whiptail lizard
	n01687978	:	372	agama
	n01688243	:	373	frilled lizard, Chlamydosaurus kingi
	n01689811	:	374	alligator lizard
	n01692333	:	375	Gila monster, Heloderma suspectum
	n01693334	:	376	green lizard, Lacerta viridis
	n01694178	:	377	African chameleon, Chamaeleo chamaeleon
	n01695060	:	378	Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis
	n01704323	:	379	triceratops
	n01697457	:	380	African crocodile, Nile crocodile, Crocodylus niloticus
	n01698640	:	381	American alligator, Alligator mississipiensis
	n01728572	:	382	thunder snake, worm snake, Carphophis amoenus
	n01728920	:	383	ringneck snake, ring-necked snake, ring snake
	n01729322	:	384	hognose snake, puff adder, sand viper
	n01729977	:	385	green snake, grass snake
	n01734418	:	386	king snake, kingsnake
	n01735189	:	387	garter snake, grass snake
	n01737021	:	388	water snake
	n01739381	:	389	vine snake
	n01740131	:	390	night snake, Hypsiglena torquata
	n01742172	:	391	boa constrictor, Constrictor constrictor
	n01744401	:	392	rock python, rock snake, Python sebae
	n01748264	:	393	Indian cobra, Naja naja
	n01749939	:	394	green mamba
	n01751748	:	395	sea snake
	n01753488	:	396	horned viper, cerastes, sand viper, horned asp, Cerastes cornutus
	n04326547	:	397	stone wall
	n01756291	:	398	sidewinder, horned rattlesnake, Crotalus cerastes
	n01629819	:	399	European fire salamander, Salamandra salamandra
	n01630670	:	400	common newt, Triturus vulgaris
	n01631663	:	401	eft
	n01632458	:	402	spotted salamander, Ambystoma maculatum
	n01632777	:	403	axolotl, mud puppy, Ambystoma mexicanum
	n01641577	:	404	bullfrog, Rana catesbeiana
	n01644373	:	405	tree frog, tree-frog
	n01644900	:	406	tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui
	n04579432	:	407	whistle
	n04592741	:	408	wing
	n03876231	:	409	paintbrush
	n03868863	:	410	oxygen mask
	n04251144	:	411	snorkel
	n03691459	:	412	loudspeaker, speaker, speaker unit, loudspeaker system, speaker system
	n03759954	:	413	microphone, mike
	n04152593	:	414	screen, CRT screen
	n03793489	:	415	mouse, computer mouse
	n03271574	:	416	electric fan, blower
	n03843555	:	417	oil filter
	n04332243	:	418	strainer
	n04265275	:	419	space heater
	n04330267	:	420	stove
	n03467068	:	421	guillotine
	n02794156	:	422	barometer
	n04118776	:	423	rule, ruler
	n03841143	:	424	odometer, hodometer, mileometer, milometer
	n04141975	:	425	scale, weighing machine
	n02708093	:	426	analog clock
	n03196217	:	427	digital clock
	n04548280	:	428	wall clock
	n03544143	:	429	hourglass
	n04355338	:	430	sundial
	n03891332	:	431	parking meter
	n04328186	:	432	stopwatch, stop watch
	n03197337	:	433	digital watch
	n04317175	:	434	stethoscope
	n04376876	:	435	syringe
	n03706229	:	436	magnetic compass
	n02841315	:	437	binoculars, field glasses, opera glasses
	n04009552	:	438	projector
	n04356056	:	439	sunglasses, dark glasses, shades
	n03692522	:	440	loupe, jeweler's loupe
	n04044716	:	441	radio telescope, radio reflector
	n02879718	:	442	bow
	n02950826	:	443	cannon
	n02749479	:	444	assault rifle, assault gun
	n04090263	:	445	rifle
	n04008634	:	446	projectile, missile
	n03085013	:	447	computer keyboard, keypad
	n04505470	:	448	typewriter keyboard
	n03126707	:	449	crane
	n03666591	:	450	lighter, light, igniter, ignitor
	n02666196	:	451	abacus
	n02977058	:	452	cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM
	n04238763	:	453	slide rule, slipstick
	n03180011	:	454	desktop computer
	n03485407	:	455	hand-held computer, hand-held microcomputer
	n03832673	:	456	notebook, notebook computer
	n03874599	:	457	padlock
	n03496892	:	458	harvester, reaper
	n04428191	:	459	thresher, thrasher, threshing machine
	n04004767	:	460	printer
	n04243546	:	461	slot, one-armed bandit
	n04525305	:	462	vending machine
	n04179913	:	463	sewing machine
	n03602883	:	464	joystick
	n04372370	:	465	switch, electric switch, electrical switch
	n03532672	:	466	hook, claw
	n02974003	:	467	car wheel
	n03874293	:	468	paddlewheel, paddle wheel
	n03944341	:	469	pinwheel
	n03992509	:	470	potter's wheel
	n03425413	:	471	gas pump, gasoline pump, petrol pump, island dispenser
	n02966193	:	472	carousel, carrousel, merry-go-round, roundabout, whirligig
	n04371774	:	473	swing
	n04067472	:	474	reel
	n04040759	:	475	radiator
	n04019541	:	476	puck, hockey puck
	n03492542	:	477	hard disc, hard disk, fixed disk
	n04355933	:	478	sunglass
	n03929660	:	479	pick, plectrum, plectron
	n02965783	:	480	car mirror
	n04258138	:	481	solar dish, solar collector, solar furnace
	n04074963	:	482	remote control, remote
	n03208938	:	483	disk brake, disc brake
	n02910353	:	484	buckle
	n03476684	:	485	hair slide
	n03627232	:	486	knot
	n03075370	:	487	combination lock
	n06359193	:	488	web site, website, internet site, site
	n03804744	:	489	nail
	n04127249	:	490	safety pin
	n04153751	:	491	screw
	n03803284	:	492	muzzle
	n04162706	:	493	seat belt, seatbelt
	n04228054	:	494	ski
	n02948072	:	495	candle, taper, wax light
	n03590841	:	496	jack-o'-lantern
	n04286575	:	497	spotlight, spot
	n04456115	:	498	torch
	n03814639	:	499	neck brace
	n03933933	:	500	pier
	n04485082	:	501	tripod
	n03733131	:	502	maypole
	n03483316	:	503	hand blower, blow dryer, blow drier, hair dryer, hair drier
	n03794056	:	504	mousetrap
	n04275548	:	505	spider web, spider's web
	n01768244	:	506	trilobite
	n01770081	:	507	harvestman, daddy longlegs, Phalangium opilio
	n01770393	:	508	scorpion
	n01773157	:	509	black and gold garden spider, Argiope aurantia
	n01773549	:	510	barn spider, Araneus cavaticus
	n01773797	:	511	garden spider, Aranea diademata
	n01774384	:	512	black widow, Latrodectus mactans
	n01774750	:	513	tarantula
	n01775062	:	514	wolf spider, hunting spider
	n01776313	:	515	tick
	n01784675	:	516	centipede
	n01990800	:	517	isopod
	n01978287	:	518	Dungeness crab, Cancer magister
	n01978455	:	519	rock crab, Cancer irroratus
	n01980166	:	520	fiddler crab
	n01981276	:	521	king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica
	n01983481	:	522	American lobster, Northern lobster, Maine lobster, Homarus americanus
	n01984695	:	523	spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish
	n01985128	:	524	crayfish, crawfish, crawdad, crawdaddy
	n01986214	:	525	hermit crab
	n02165105	:	526	tiger beetle
	n02165456	:	527	ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle
	n02167151	:	528	ground beetle, carabid beetle
	n02168699	:	529	long-horned beetle, longicorn, longicorn beetle
	n02169497	:	530	leaf beetle, chrysomelid
	n02172182	:	531	dung beetle
	n02174001	:	532	rhinoceros beetle
	n02177972	:	533	weevil
	n02190166	:	534	fly
	n02206856	:	535	bee
	n02226429	:	536	grasshopper, hopper
	n02229544	:	537	cricket
	n02231487	:	538	walking stick, walkingstick, stick insect
	n02233338	:	539	cockroach, roach
	n02236044	:	540	mantis, mantid
	n02256656	:	541	cicada, cicala
	n02259212	:	542	leafhopper
	n02264363	:	543	lacewing, lacewing fly
	n02268443	:	544	dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk
	n02268853	:	545	damselfly
	n02276258	:	546	admiral
	n02277742	:	547	ringlet, ringlet butterfly
	n02279972	:	548	monarch, monarch butterfly, milkweed butterfly, Danaus plexippus
	n02280649	:	549	cabbage butterfly
	n02281406	:	550	sulphur butterfly, sulfur butterfly
	n02281787	:	551	lycaenid, lycaenid butterfly
	n01910747	:	552	jellyfish
	n01914609	:	553	sea anemone, anemone
	n01917289	:	554	brain coral
	n01924916	:	555	flatworm, platyhelminth
	n01930112	:	556	nematode, nematode worm, roundworm
	n01943899	:	557	conch
	n01944390	:	558	snail
	n01945685	:	559	slug
	n01950731	:	560	sea slug, nudibranch
	n01955084	:	561	chiton, coat-of-mail shell, sea cradle, polyplacophore
	n02319095	:	562	sea urchin
	n02321529	:	563	sea cucumber, holothurian
	n03584829	:	564	iron, smoothing iron
	n03297495	:	565	espresso maker
	n03761084	:	566	microwave, microwave oven
	n03259280	:	567	Dutch oven
	n04111531	:	568	rotisserie
	n04442312	:	569	toaster
	n04542943	:	570	waffle iron
	n04517823	:	571	vacuum, vacuum cleaner
	n03207941	:	572	dishwasher, dish washer, dishwashing machine
	n04070727	:	573	refrigerator, icebox
	n04554684	:	574	washer, automatic washer, washing machine
	n03133878	:	575	Crock Pot
	n03400231	:	576	frying pan, frypan, skillet
	n04596742	:	577	wok
	n02939185	:	578	caldron, cauldron
	n03063689	:	579	coffeepot
	n04398044	:	580	teapot
	n04270147	:	581	spatula
	n02699494	:	582	altar
	n04486054	:	583	triumphal arch
	n03899768	:	584	patio, terrace
	n04311004	:	585	steel arch bridge
	n04366367	:	586	suspension bridge
	n04532670	:	587	viaduct
	n02793495	:	588	barn
	n03457902	:	589	greenhouse, nursery, glasshouse
	n03877845	:	590	palace
	n03781244	:	591	monastery
	n03661043	:	592	library
	n02727426	:	593	apiary, bee house
	n02859443	:	594	boathouse
	n03028079	:	595	church, church building
	n03788195	:	596	mosque
	n04346328	:	597	stupa, tope
	n03956157	:	598	planetarium
	n04081281	:	599	restaurant, eating house, eating place, eatery
	n03032252	:	600	cinema, movie theater, movie theatre, movie house, picture palace
	n03529860	:	601	home theater, home theatre
	n03697007	:	602	lumbermill, sawmill
	n03065424	:	603	coil, spiral, volute, whorl, helix
	n03837869	:	604	obelisk
	n04458633	:	605	totem pole
	n02980441	:	606	castle
	n04005630	:	607	prison, prison house
	n03461385	:	608	grocery store, grocery, food market, market
	n02776631	:	609	bakery, bakeshop, bakehouse
	n02791270	:	610	barbershop
	n02871525	:	611	bookshop, bookstore, bookstall
	n02927161	:	612	butcher shop, meat market
	n03089624	:	613	confectionery, confectionary, candy store
	n04200800	:	614	shoe shop, shoe-shop, shoe store
	n04443257	:	615	tobacco shop, tobacconist shop, tobacconist
	n04462240	:	616	toyshop
	n03388043	:	617	fountain
	n03042490	:	618	cliff dwelling
	n04613696	:	619	yurt
	n03216828	:	620	dock, dockage, docking facility
	n02892201	:	621	brass, memorial tablet, plaque
	n03743016	:	622	megalith, megalithic structure
	n02788148	:	623	bannister, banister, balustrade, balusters, handrail
	n02894605	:	624	breakwater, groin, groyne, mole, bulwark, seawall, jetty
	n03160309	:	625	dam, dike, dyke
	n03000134	:	626	chainlink fence
	n03930313	:	627	picket fence, paling
	n04604644	:	628	worm fence, snake fence, snake-rail fence, Virginia fence
	n01755581	:	629	diamondback, diamondback rattlesnake, Crotalus adamanteus
	n03459775	:	630	grille, radiator grille
	n04239074	:	631	sliding door
	n04501370	:	632	turnstile
	n03792972	:	633	mountain tent
	n04149813	:	634	scoreboard
	n03530642	:	635	honeycomb
	n03961711	:	636	plate rack
	n03903868	:	637	pedestal, plinth, footstall
	n02814860	:	638	beacon, lighthouse, beacon light, pharos
	n01665541	:	639	leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea
	n07711569	:	640	mashed potato
	n07720875	:	641	bell pepper
	n07714571	:	642	head cabbage
	n07714990	:	643	broccoli
	n07715103	:	644	cauliflower
	n07716358	:	645	zucchini, courgette
	n07716906	:	646	spaghetti squash
	n07717410	:	647	acorn squash
	n07717556	:	648	butternut squash
	n07718472	:	649	cucumber, cuke
	n07718747	:	650	artichoke, globe artichoke
	n07730033	:	651	cardoon
	n07734744	:	652	mushroom
	n04209239	:	653	shower curtain
	n03594734	:	654	jean, blue jean, denim
	n02971356	:	655	carton
	n03485794	:	656	handkerchief, hankie, hanky, hankey
	n04133789	:	657	sandal
	n02747177	:	658	ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin
	n04125021	:	659	safe
	n07579787	:	660	plate
	n03814906	:	661	necklace
	n03134739	:	662	croquet ball
	n03404251	:	663	fur coat
	n04423845	:	664	thimble
	n03877472	:	665	pajama, pyjama, pj's, jammies
	n04120489	:	666	running shoe
	n03838899	:	667	oboe, hautboy, hautbois
	n03062245	:	668	cocktail shaker
	n03014705	:	669	chest
	n03717622	:	670	manhole cover
	n03777754	:	671	modem
	n04493381	:	672	tub, vat
	n04476259	:	673	tray
	n02777292	:	674	balance beam, beam
	n07693725	:	675	bagel, beigel
	n04536866	:	676	violin, fiddle
	n03998194	:	677	prayer rug, prayer mat
	n03617480	:	678	kimono
	n07590611	:	679	hot pot, hotpot
	n04579145	:	680	whiskey jug
	n03623198	:	681	knee pad
	n07248320	:	682	book jacket, dust cover, dust jacket, dust wrapper
	n04277352	:	683	spindle
	n04229816	:	684	ski mask
	n02823428	:	685	beer bottle
	n03127747	:	686	crash helmet
	n02877765	:	687	bottlecap
	n04435653	:	688	tile roof
	n03724870	:	689	mask
	n03710637	:	690	maillot
	n03920288	:	691	Petri dish
	n03379051	:	692	football helmet
	n02807133	:	693	bathing cap, swimming cap
	n04399382	:	694	teddy, teddy bear
	n03527444	:	695	holster
	n03983396	:	696	pop bottle, soda bottle
	n03924679	:	697	photocopier
	n04532106	:	698	vestment
	n06785654	:	699	crossword puzzle, crossword
	n03445777	:	700	golf ball
	n07613480	:	701	trifle
	n04350905	:	702	suit, suit of clothes
	n04562935	:	703	water tower
	n03325584	:	704	feather boa, boa
	n03045698	:	705	cloak
	n07892512	:	706	red wine
	n03250847	:	707	drumstick
	n04192698	:	708	shield, buckler
	n03026506	:	709	Christmas stocking
	n03534580	:	710	hoopskirt, crinoline
	n07565083	:	711	menu
	n04296562	:	712	stage
	n02869837	:	713	bonnet, poke bonnet
	n07871810	:	714	meat loaf, meatloaf
	n02799071	:	715	baseball
	n03314780	:	716	face powder
	n04141327	:	717	scabbard
	n04357314	:	718	sunscreen, sunblock, sun blocker
	n02823750	:	719	beer glass
	n13052670	:	720	hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa
	n07583066	:	721	guacamole
	n04599235	:	722	wool, woolen, woollen
	n07802026	:	723	hay
	n02883205	:	724	bow tie, bow-tie, bowtie
	n03709823	:	725	mailbag, postbag
	n04560804	:	726	water jug
	n02909870	:	727	bucket, pail
	n03207743	:	728	dishrag, dishcloth
	n04263257	:	729	soup bowl
	n07932039	:	730	eggnog
	n03786901	:	731	mortar
	n04479046	:	732	trench coat
	n03873416	:	733	paddle, boat paddle
	n02999410	:	734	chain
	n04367480	:	735	swab, swob, mop
	n03775546	:	736	mixing bowl
	n07875152	:	737	potpie
	n04591713	:	738	wine bottle
	n04201297	:	739	shoji
	n02916936	:	740	bulletproof vest
	n03240683	:	741	drilling platform, offshore rig
	n02840245	:	742	binder, ring-binder
	n02963159	:	743	cardigan
	n04370456	:	744	sweatshirt
	n03991062	:	745	pot, flowerpot
	n02843684	:	746	birdhouse
	n03599486	:	747	jinrikisha, ricksha, rickshaw
	n03482405	:	748	hamper
	n03942813	:	749	ping-pong ball
	n03908618	:	750	pencil box, pencil case
	n07584110	:	751	consomme
	n02730930	:	752	apron
	n04023962	:	753	punching bag, punch bag, punching ball, punchball
	n02769748	:	754	backpack, back pack, knapsack, packsack, rucksack, haversack
	n10148035	:	755	groom, bridegroom
	n02817516	:	756	bearskin, busby, shako
	n03908714	:	757	pencil sharpener
	n02906734	:	758	broom
	n02667093	:	759	abaya
	n03787032	:	760	mortarboard
	n03980874	:	761	poncho
	n03141823	:	762	crutch
	n03976467	:	763	Polaroid camera, Polaroid Land camera
	n04264628	:	764	space bar
	n07930864	:	765	cup
	n04039381	:	766	racket, racquet
	n06874185	:	767	traffic light, traffic signal, stoplight
	n04033901	:	768	quill, quill pen
	n04041544	:	769	radio, wireless
	n02128757	:	770	snow leopard, ounce, Panthera uncia
	n07860988	:	771	dough
	n03146219	:	772	cuirass
	n03763968	:	773	military uniform
	n03676483	:	774	lipstick, lip rouge
	n04209133	:	775	shower cap
	n03782006	:	776	monitor
	n03857828	:	777	oscilloscope, scope, cathode-ray oscilloscope, CRO
	n03775071	:	778	mitten
	n02892767	:	779	brassiere, bra, bandeau
	n07684084	:	780	French loaf
	n04522168	:	781	vase
	n03764736	:	782	milk can
	n04118538	:	783	rugby ball
	n03887697	:	784	paper towel
	n13044778	:	785	earthstar
	n03291819	:	786	envelope
	n03770439	:	787	miniskirt, mini
	n03124170	:	788	cowboy hat, ten-gallon hat
	n04487081	:	789	trolleybus, trolley coach, trackless trolley
	n03916031	:	790	perfume, essence
	n02808440	:	791	bathtub, bathing tub, bath, tub
	n07697537	:	792	hotdog, hot dog, red hot
	n12985857	:	793	coral fungus
	n02917067	:	794	bullet train, bullet
	n03938244	:	795	pillow
	n15075141	:	796	toilet tissue, toilet paper, bathroom tissue
	n02978881	:	797	cassette
	n02966687	:	798	carpenter's kit, tool kit
	n03633091	:	799	ladle
	n13040303	:	800	stinkhorn, carrion fungus
	n03690938	:	801	lotion
	n03476991	:	802	hair spray
	n02669723	:	803	academic gown, academic robe, judge's robe
	n03220513	:	804	dome
	n03127925	:	805	crate
	n04584207	:	806	wig
	n07880968	:	807	burrito
	n03937543	:	808	pill bottle
	n03000247	:	809	chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour
	n04418357	:	810	theater curtain, theatre curtain
	n04590129	:	811	window shade
	n02795169	:	812	barrel, cask
	n04553703	:	813	washbasin, handbasin, washbowl, lavabo, wash-hand basin
	n02783161	:	814	ballpoint, ballpoint pen, ballpen, Biro
	n02802426	:	815	basketball
	n02808304	:	816	bath towel
	n03124043	:	817	cowboy boot
	n03450230	:	818	gown
	n04589890	:	819	window screen
	n12998815	:	820	agaric
	n02113799	:	821	standard poodle
	n02992529	:	822	cellular telephone, cellular phone, cellphone, cell, mobile phone
	n03825788	:	823	nipple
	n02790996	:	824	barbell
	n03710193	:	825	mailbox, letter box
	n03630383	:	826	lab coat, laboratory coat
	n03347037	:	827	fire screen, fireguard
	n03769881	:	828	minibus
	n03871628	:	829	packet
	n02132136	:	830	brown bear, bruin, Ursus arctos
	n03976657	:	831	pole
	n03535780	:	832	horizontal bar, high bar
	n04259630	:	833	sombrero
	n03929855	:	834	pickelhaube
	n04049303	:	835	rain barrel
	n04548362	:	836	wallet, billfold, notecase, pocketbook
	n02979186	:	837	cassette player
	n06596364	:	838	comic book
	n03935335	:	839	piggy bank, penny bank
	n06794110	:	840	street sign
	n02825657	:	841	bell cote, bell cot
	n03388183	:	842	fountain pen
	n04591157	:	843	Windsor tie
	n04540053	:	844	volleyball
	n03866082	:	845	overskirt
	n04136333	:	846	sarong
	n04026417	:	847	purse
	n02865351	:	848	bolo tie, bolo, bola tie, bola
	n02834397	:	849	bib
	n03888257	:	850	parachute, chute
	n04235860	:	851	sleeping bag
	n04404412	:	852	television, television system
	n04371430	:	853	swimming trunks, bathing trunks
	n03733805	:	854	measuring cup
	n07920052	:	855	espresso
	n07873807	:	856	pizza, pizza pie
	n02895154	:	857	breastplate, aegis, egis
	n04204238	:	858	shopping basket
	n04597913	:	859	wooden spoon
	n04131690	:	860	saltshaker, salt shaker
	n07836838	:	861	chocolate sauce, chocolate syrup
	n09835506	:	862	ballplayer, baseball player
	n03443371	:	863	goblet
	n13037406	:	864	gyromitra
	n04336792	:	865	stretcher
	n04557648	:	866	water bottle
	n02445715	:	867	skunk, polecat, wood pussy
	n04254120	:	868	soap dispenser
	n03595614	:	869	jersey, T-shirt, tee shirt
	n04146614	:	870	school bus
	n03598930	:	871	jigsaw puzzle
	n03958227	:	872	plastic bag
	n04069434	:	873	reflex camera
	n03188531	:	874	diaper, nappy, napkin
	n02786058	:	875	Band Aid
	n07615774	:	876	ice lolly, lolly, lollipop, popsicle
	n04525038	:	877	velvet
	n04409515	:	878	tennis ball
	n03424325	:	879	gasmask, respirator, gas helmet
	n03223299	:	880	doormat, welcome mat
	n03680355	:	881	Loafer
	n07614500	:	882	ice cream, icecream
	n07695742	:	883	pretzel
	n04033995	:	884	quilt, comforter, comfort, puff
	n03710721	:	885	maillot, tank suit
	n04392985	:	886	tape player
	n03047690	:	887	clog, geta, patten, sabot
	n03584254	:	888	iPod
	n13054560	:	889	bolete
	n02138441	:	890	meerkat, mierkat
	n10565667	:	891	scuba diver
	n03950228	:	892	pitcher, ewer
	n03729826	:	893	matchstick
	n02837789	:	894	bikini, two-piece
	n04254777	:	895	sock
	n02988304	:	896	CD player
	n03657121	:	897	lens cap, lens cover
	n04417672	:	898	thatch, thatched roof
	n04523525	:	899	vault
	n02815834	:	900	beaker
	n09229709	:	901	bubble
	n07697313	:	902	cheeseburger
	n03888605	:	903	parallel bars, bars
	n03355925	:	904	flagpole, flagstaff
	n03063599	:	905	coffee mug
	n04116512	:	906	rubber eraser, rubber, pencil eraser
	n04325704	:	907	stole
	n07831146	:	908	carbonara
	n03255030	:	909	dumbbell
        
Synsets new in ILSVRC2012
	n02110185	:	910	Siberian husky
	n02102040	:	911	English springer, English springer spaniel
	n02110063	:	912	malamute, malemute, Alaskan malamute
	n02089867	:	913	Walker hound, Walker foxhound
	n02102177	:	914	Welsh springer spaniel
	n02091134	:	915	whippet
	n02092339	:	916	Weimaraner
	n02098105	:	917	soft-coated wheaten terrier
	n02096437	:	918	Dandie Dinmont, Dandie Dinmont terrier
	n02105641	:	919	Old English sheepdog, bobtail
	n02091635	:	920	otterhound, otter hound
	n02088466	:	921	bloodhound, sleuthhound
	n02096051	:	922	Airedale, Airedale terrier
	n02097130	:	923	giant schnauzer
	n02089078	:	924	black-and-tan coonhound
	n02086910	:	925	papillon
	n02113978	:	926	Mexican hairless
	n02113186	:	927	Cardigan, Cardigan Welsh corgi
	n02105162	:	928	malinois
	n02098413	:	929	Lhasa, Lhasa apso
	n02091467	:	930	Norwegian elkhound, elkhound
	n02106550	:	931	Rottweiler
	n02091831	:	932	Saluki, gazelle hound
	n02104365	:	933	schipperke
	n02112706	:	934	Brabancon griffon
	n02098286	:	935	West Highland white terrier
	n02095889	:	936	Sealyham terrier, Sealyham
	n02090721	:	937	Irish wolfhound
	n02108000	:	938	EntleBucher
	n02108915	:	939	French bulldog
	n02107683	:	940	Bernese mountain dog
	n02085936	:	941	Maltese dog, Maltese terrier, Maltese
	n02094114	:	942	Norfolk terrier
	n02087046	:	943	toy terrier
	n02096177	:	944	cairn, cairn terrier
	n02105056	:	945	groenendael
	n02101556	:	946	clumber, clumber spaniel
	n02088094	:	947	Afghan hound, Afghan
	n02085782	:	948	Japanese spaniel
	n02090622	:	949	borzoi, Russian wolfhound
	n02113624	:	950	toy poodle
	n02093859	:	951	Kerry blue terrier
	n02097298	:	952	Scotch terrier, Scottish terrier, Scottie
	n02096585	:	953	Boston bull, Boston terrier
	n02107574	:	954	Greater Swiss Mountain dog
	n02107908	:	955	Appenzeller
	n02086240	:	956	Shih-Tzu
	n02102973	:	957	Irish water spaniel
	n02112018	:	958	Pomeranian
	n02093647	:	959	Bedlington terrier
	n02097047	:	960	miniature schnauzer
	n02106030	:	961	collie
	n02093991	:	962	Irish terrier
	n02110627	:	963	affenpinscher, monkey pinscher, monkey dog
	n02097658	:	964	silky terrier, Sydney silky
	n02088364	:	965	beagle
	n02111129	:	966	Leonberg
	n02100236	:	967	German short-haired pointer
	n02115913	:	968	dhole, Cuon alpinus
	n02099849	:	969	Chesapeake Bay retriever
	n02108422	:	970	bull mastiff
	n02104029	:	971	kuvasz
	n02110958	:	972	pug, pug-dog
	n02099429	:	973	curly-coated retriever
	n02094258	:	974	Norwich terrier
	n02112350	:	975	keeshond
	n02095570	:	976	Lakeland terrier
	n02097209	:	977	standard schnauzer
	n02097474	:	978	Tibetan terrier, chrysanthemum dog
	n02095314	:	979	wire-haired fox terrier
	n02088238	:	980	basset, basset hound
	n02112137	:	981	chow, chow chow
	n02093428	:	982	American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier
	n02105855	:	983	Shetland sheepdog, Shetland sheep dog, Shetland
	n02111500	:	984	Great Pyrenees
	n02085620	:	985	Chihuahua
	n02099712	:	986	Labrador retriever
	n02111889	:	987	Samoyed, Samoyede
	n02088632	:	988	bluetick
	n02105412	:	989	kelpie
	n02107312	:	990	miniature pinscher
	n02091032	:	991	Italian greyhound
	n02102318	:	992	cocker spaniel, English cocker spaniel, cocker
	n02102480	:	993	Sussex spaniel
	n02113023	:	994	Pembroke, Pembroke Welsh corgi
	n02086646	:	995	Blenheim spaniel
	n02091244	:	996	Ibizan hound, Ibizan Podenco
	n02089973	:	997	English foxhound
	n02105251	:	998	briard
	n02093754	:	999	Border terrier
