# coding: UTF-8
import csv
import numpy as np
import pandas as pd
import MeCab as mc
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import math
import itertools
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def idf(s, l):
    idf_v = 0
    for i in range(len(l)):
        if s in l[i]:
            idf_v += 1
    return np.log(len(l) / (idf_v + 1)) + 1

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1

def out_symbol(x):
    symbols = ["・", "，", ",", ".", "、", "。", "(", ")", "（", "）", "「", "」", "ー", "-", "*"]
    return x not in symbols

train_data = np.loadtxt('needs_scan_panel_training.csv', delimiter=',')
test_data = np.loadtxt('needs_scan_panel_test.csv', delimiter=',')

user_n = train_data.shape[0]
item_n = train_data.shape[1]

# 正しい値のリスト
correct_value = np.zeros((len(test_data)))
train_data_sparse = deepcopy(train_data)

# テストデータの部分を−１に置換
# 正しい値を保存
for i in range(test_data.shape[0]):
    train_data_sparse[int(test_data[i][0])][int(test_data[i][1])] = -1
    correct_value[i] = test_data[i][2]

# content = ["レールや架線を用いないでエンジンによって陸上を移動できる車のうち，原動機付自転車以外のものをいう。一般には用途によって乗用自動車 (バス，一般用自動車) ，乗貨兼用自動車，貨物自動車，特殊自動車など，また，法規による分類では自動二輪車，自動三輪車，普通自動車，軽自動車，大型自動車，小型特殊自動車，大型特殊自動車に，エンジンによってガソリン車，ディーゼル車などに分類される。自動車は 18世紀に蒸気機関を動力源として試みられたのが最初で，19世紀末にドイツでガソリン機関がつくられてからガソリン車を主体として発展した。 20世紀末から，排出ガスによる大気汚染を防止する電気自動車，ハイブリッド車など，新しい動力源をもつ自動車の開発が進んでいる。"
# , "打弦楽器の一種。現在の標準型は 88の鍵盤と，音の持続と強弱を調整する2～3個のペダルがついて，鍵盤と連動するハンマーによって弦を打ち音を出す。平型のグランド・ピアノと竪型のアップライト・ピアノがある。チェンバロに似た型をとっているが，発音の機構は弦をハンマーで打って音を出す古代の楽器ダルシマーに近い。 1709年にフィレンツェの B.クリストフォリが，強弱を自由に出せるチェンバロとして，グラベチェンバロ・コル・ピアノ・エ・フォルテという名で発明し，以後ピアノフォルテ，またはピアノと呼ばれて，18～19世紀に多くの改良がなされ現代のものになった。"
# , "映像・音声情報の電子信号を磁気テープに記録し，その情報を再生する電気機器。記録方式にはトランスバース式（4ヘッド）とヘリカルスキャン式の 2種類があり，テープの形状にはオープンリール式とカセット式がある。トランスバース式装置は，2インチ（5cm）テープに直交する軸上で 4個の磁気ヘッドが回転する。ヘッドとテープの相対速度は毎分 3800cmに達し，高画質を実現できる。放送業界の要望に合わせ，音声トラック，コントロールトラック，キュートラックは縦方向に記録される。アメリカテレビジョン標準方式委員会の規格に準じ，電子ビームが 525本の水平走査線を毎秒 60回走査する。ヘリカルスキャン式装置は家庭・個人向け仕様で，0.5インチまたは 0.75インチのテープがドラムの周囲を螺旋状に走行する。カセット式のテープを用いるビデオカセットレコーダは 1960年代に原型が開発され，1969年にソニーが比較的使いやすい安価な機種を発表した。1970年代にソニーがベータマックス方式，松下電器産業（→パナソニック）が VHS方式を開発し，一般家庭が購入できる手頃な価格になった。VHSとベータマックスはいずれも 0.5インチ幅のテープを用いるが互換性はない。1985年初めには 0.3インチ（8mm）幅のテープを使用する方式も開発された。テレビジョンの番組の録画，再生に用いられたほか，録画済みカセットも市販されたが，21世紀にはビデオテープに代わって，デジタル信号を磁気ディスクや光学ディスクに記録する方式が主流となった。"
# , "室内の空気調和を行う装置。単にエアコンともいう。人間の活動にもっとも適した温度・湿度・気流分布に調節し、同時に空気中の塵埃(じんあい)を除く役目をする。近年は冷房だけでなく、除湿、暖房を含めて年間を通しての空気調和の役割を果たし、普及が目覚ましい。また、低騒音化、省エネルギー化が進み、エレクトロニクス技術の導入が積極的に行われている。原理は、冷凍機や冷蔵庫などと同様に、液体が気化するときの気化熱を利用したもので、冷媒には、低温でも気化しやすい液体を用い、一般にフロンが使われていた。しかし、フロンはオゾン層を破壊するため使用禁止となり、代替フロンのHFC（ハイドロフルオロカーボン）などが用いられるようになった。その代替フロンも温室効果をもたらすことが判明し、さらにノンフロン系の冷媒の開発が進められていて、ノンフロンのルームエアコンも登場している。構造は、密封された鉄製の容器の中にモーターと圧縮機が直結され、モーターによって圧縮機を回して冷媒を圧縮し、高温・高圧にして凝縮器に送る。圧縮機には往復運動式（レシプロ式）とロータリー式とがあるが、最近は効率もよく、小型・軽量のロータリー式が多く使われている。凝縮器は銅パイプの表面にアルミニウムのフィンを取り付けたもので、高温・高圧の冷媒を放熱させ、気体から液体に冷却・液化する働きをする。毛細管は、凝縮器からの高圧の液状冷媒を、蒸発器で蒸発しやすいように圧力を下げて、低温・低圧の湿り気体（気体と液体とが混在した状態）にする装置で、内径1～2ミリメートル程度の細い銅パイプである。蒸発器の構造は凝縮器とほぼ同様で、圧縮された冷媒は凝縮器、毛細管を経てここで蒸発し、周囲から熱を奪う。したがって、蒸発器の表面に接触した空気の温度は下がり、空気中の水分は蒸発器の表面で露点以下となり、水滴となって除かれる。送風機関係は、能率よく熱交換を行わせるために強制的に空気を送る。ルームエアコンには、ルームクーラーといわれる冷房専用型とヒートポンプ型（冷房・暖房兼用型）とがある。ヒートポンプ型の基本構造は、冷房専用タイプと変わりなく、蒸発器と凝縮器を逆にすれば、冷房が暖房となる。ヒートポンプ型の切り換え弁を四方弁という。ヒートポンプ型の特徴は、温度の低いところにある熱量を、温度を高めて使うことができる点にある。また、その際、消費した仕事（たとえば電力）より大きい熱量が得られる。電力をそのまま電熱として暖房するとすれば、1キロワット時で860キロカロリー毎時の熱量にしかならないが、成績係数3のヒートポンプを用いれば、3倍の2580キロカロリー毎時の熱量として利用できる。据え付け方法には、本体が一体化されて窓や壁に取り付け、キャビネットの後部を室外に出し、凝縮器を外気により冷却するウィンド型と、部屋の内側に蒸発器と送風機を収めたユニットを置き、外側に圧縮機、凝縮器および送風機を収めたユニットを置いて、この間を冷媒を通すパイプで結んでいるセパレート型とがある。ウィンド型は、小容量機種が主で、縦方向に長い細幅タイプが多く、戸外に面した外壁や窓の構造がルームエアコンの重量に耐える場所に据え付ける。セパレート型は、室内ユニットを壁に掛ける壁掛けタイプ、床に置く床置きタイプ、天井から吊(つ)り下げる天井吊り下げタイプなどがある。セパレート型は圧縮機が室外ユニットにあるため、室内騒音が小さく、また、壁に冷媒配管と配線を通す小さな穴をあけるだけで取り付けられるので、現在ではルームエアコンの主流となっている。"
# , "コンピュータの分類の一つを表し、個人所有のコンピュータをさすが、明確な定義はなく、オフィスで使用するコンピュータをさす場合もある。日本では縮めてパソコンという。欧米では略してPCといい、日本でもPCと略す場合もある。パーソナルコンピュータという語は、1970年代初め、アメリカのIBM社とヒューレット・パッカード社が発売したコンピュータに対して使われたのが最初である。普及当初は、マイクロプロセッサーや記憶装置のメモリーなどをキーボードの筐体(きょうたい)（平たい箱型）の中に組み込み、これに家庭用テレビを表示装置（ディスプレー）として接続することで1個のコンピュータとして使えるようにしたものが多かった。その後、電子回路技術の発展によってパーソナルコンピュータの性能は飛躍的に向上した。とくに、IBM社が採用したインテル社のマイクロプロセッサーを用いたハードウェアの基本設計と、オペレーティングシステムoperating system（略称OS）としてのマイクロソフト社のDOS（ドス、disk OSの略称）の組合せがデファクト標準（市場での競争を通して決まる事実上の標準）として広く使われるに至った。この流れとは別に、1984年にマッキントッシュMacintoshという名で個人用の知的文房具として位置づけたPCシステムを開発したのがアメリカのアップル・コンピュータ（現アップル）社であった。当時としては先進的な、マウスで画面上のアイコンをクリックするだけで簡単にソフトウェアが使える仕組みGUI（グイともいう。graphical user interfaceの略称）を基本とするOSをのせていた。いまではマイクロソフト社のOSもウィンドウズWindowsと名を変え、全面的にGUIを取り入れたものとなっている。パーソナルコンピュータは、OSの発展や低価格化、インターネットの爆発的な発展などと相まって急速に普及した。その形態も、机上で使うデスクトップ型ばかりでなく、携帯可能なラップトップ型、さらに小型のノート型と多岐にわたり、情報技術information technology（略称IT）の発達を背景に急速に展開する情報化社会を支える基盤を構成するに至っている。"
# , "文書を効率よく作成するための機器またはプログラム。文書の入出力や記憶をし、これを改訂、編集することができる。略してワープロということが多い。汎用(はんよう)のコンピュータを利用するものと専用機とがある。タイプライターを日常的に使っている欧米では、コンピュータのプログラム作成用に開発された会話型のエディター（編集系）が登場したとき、これで手紙を書いたり、原稿をつくったりするのは自然なことであった。やがて、そのような文書の作成専用のプログラムもつくられるようになった。なかでも、ソルツァーJerome H. Saltzer（1939―　）が1965年につくったランオフrunoffと、クヌースDonald Ervin Knuth（1938―　）が79年につくったテフTeXや、リードがつくったスクライブScribeは有名である。パーソナルコンピュータには、マイクロプロ社が1978年につくったWord Star以降、さまざまなものがある。現在のワードプロセッサーの多くは辞書を備えていて綴(つづ)りや文法の誤りを検出することができる。日本では、漢字の入力の方法と機器の価格が問題で開発が遅れていたが、近年、パーソナルコンピュータの普及とともに出力装置が安くなってきたので、家庭用のワードプロセッサーも普及してきた。これには、6349種の漢字にJIS(ジス)コードを制定したことが大きく寄与している。漢字の入力方式には、漢字ごとに2個程度のキーの組合せを定めておく方式と、漢字の読み方を仮名やローマ字で入力して漢字に変換する方式とがある。後者の仮名漢字（ローマ字漢字）変換方式では、漢字ごとではなく文節や文章ごとに入力するのが普通であり、一般の利用者にはこの方式が便利である。この方式は辞書を用いるので、送り仮名や当て字の誤りが防げる反面、同音異義語のなかから適切なものを選択するのに手間がかかるし、選択の誤りも生じる。この欠点は、辞書を充実して文脈を読み取ることによってだいぶ改善されてきている。入力を漢字符号に変換する部分のプログラムをFEP、IME、IMなどとよび、ワードプロセッサーから独立して他のソフトウェアの漢字入力にも用いられている。ワードプロセッサーの重要な機能の一つは、文書をHDDなど他の媒体に記憶させることで、類似の文書をつくる業務や手紙の控えなどに役だつ。もう一つの重要な機能は、文書を改訂することで、手紙や書類や各種の原稿などを気軽に清書し、修正できる効果は絶大である。この機能を十分に発揮するため、文章の入れ換えや複写、字句の探索などができるようになっている。また、よく使われる長い単語や文章を登録しておいて短い読み方で引用したり、グラフの作成や表の計算をしたりすることができるようになっているものも多い。ワードプロセッサー用ソフトウェアの高機能化に伴い、文章や表組みに加えて画像などのグラフィックスも扱えるようになっている。レイアウトの自由度も高く、出版用のDTPソフト並みの機能をもっている。おもなソフトウェアとして「ワードWord」や「一太郎」などがある。また、文字（テキスト）の入力と編集のみに特化した場合、テキストエディターが使われることが多い。Windows(ウィンドウズ)に付属する「メモ帳」などがそれにあたる。"
# , "デジタルオーディオディスクの一つ。デジタル化した音声信号を円盤に記録したレコード。略してCDともいう。信号は、ディスク表面のピットとよばれる幅0.4マイクロメートル、長さ0.8～3マイクロメートル（信号によって変わる）、深さ0.1マイクロメートルの顕微鏡で見なければわからないような小さな凹凸として記録される。これをレーザー光によって読み出し、音声信号に復原する。なお、レーザー光の反射をよくするため、ピットの裏側にアルミの反射膜がコートされている。直径12センチメートルのディスクを使用して、片面のみではあるが最大74分の演奏が収録できる。CDの特長は、従来のLPレコードに比べて、ダイナミックレンジ（音の大小の範囲）が広いこと、非接触のため何回使用してもディスクが劣化せず、音質・機能のうえで優れていることにある。また、傷やほこりに強く、小型で収納スペースが少なくてすみ、曲のランダムアクセスや早送り、逆転もできる。日本では1982年（昭和57）10月から世界に先駆けて発売された。より詳細な解説は「デジタルオーディオディスク」の項目を参照されたい。"
# , "ビデオ信号すなわち映像と音の信号をディスク（円板）状の媒体に記録したもの。これをプレーヤーにかけてディスプレー装置に接続すると映像と音を再生することができる。この装置全体をさす場合もある。DVDもビデオ信号の記録に使われ、その意味ではビデオディスクの仲間に入れることができる。事実DVDの初期にはビデオディスクとして扱われたが、DVDはビデオ信号の記録だけでなく、さまざまなデータの記録に使われ、コンピュータの周辺機器としての性格が強くなったため、現在は別扱いされることが多い。ここでは、現在の扱いに準じてDVDは別項目とし、DVDが出現する以前の、アナログ技術を使ったディスクをビデオディスクとして取り扱う。ビデオディスク以外にビデオ信号を記録する方法として、ビデオテープレコーダーがある。ビデオテープレコーダーは録画・再生の両機能をもち、テープを反復再利用したり編集したりすることができる利点があるが、すばやく任意の場所を選んで再生することが苦手であること、繰り返して使用するとテープが劣化する可能性があることなどの欠点がある。これらの欠点は、ディスク状媒体を用い非接触式の再生方法を使うことで解決される。このためさまざまな方法が開発され、提案された結果、ビデオディスクが誕生した。"
# , "2輪の自動車の総称。前後に各1輪をもち、通常、後輪を駆動、前輪を操向して走る自動車。オートバイの呼称は最初期の英語オートバイシクルautobicycle（自動自転車）の名残(なごり)で、今日では日本でしか通用しない。英語ではモーターサイクルmotorcycle、米語ではモーターバイクmotorbike、フランス語ではモトシクレットmotocyclette、イタリア語ではモトチクレッタmotocicletta、ドイツ語ではモートルラートMotorradとよび、世界的に「モト」あるいは「バイク」という呼び方が定着している。ごく一般的に二輪自動車の総称だが、日本では区別の必要上モーペットやスクーターを除外することもある。法律上は自動二輪車と50cc以下の原動機付自転車に分かれる。"
# , "自転車とは、乗員の運転操作により人力で駆動され走行する車両（日本工業規格「自転車の分類と諸元」）、または前輪と後輪の車輪を有し、一つまたは複数のサドルを備え、ペダル上の乗員の脚力で推進される車両と定義される。いずれにしても、地上を移動するために、人間の筋力がもっとも効率よく発揮されてその目的を達成できるのが自転車である。人間の筋力と技能が要求されて目的を成就することができるのが道具であるとすれば、自転車はマシンというよりも道具というべきであり、21世紀に継承される数少ない道具のなかの一つであろう。"
# , "大型の電気冷蔵庫。1913年にアメリカで実用化され、20年代に日本に輸入された。国産第1号の電気冷蔵庫は、1930年（昭和5）に発売されたが、価格は700円以上で家が一軒買えるほど高価なものであった。JIS(ジス)（日本工業規格）によれば、電気によって駆動される圧縮冷凍機によって冷却され、冷凍食品以外の食品を貯蔵するために必要な温度を保つことができる貯蔵室（冷蔵室という）が一つ以上あるものを冷蔵庫といい、冷蔵室を一つ以上もち、なおかつ冷凍食品を貯蔵するために必要な温度を保つことができる貯蔵室（冷凍室という）が一つ以上あるものを冷凍冷蔵庫といい、これらを総称したものが電気冷蔵庫である、ということになる。電気冷蔵庫の原理は、液体が気化する際の気化熱を利用したもので、これに使用する液体（冷媒）は、一般にフロンガス（ジクロロジフルオロメタン）が使用されていたが、オゾン層破壊が問題となり、使用禁止となったため、代替フロンのHFC（ハイドロフルオロカーボン）などが用いられている。この代替フロンもきわめて高い温室効果をもたらす物質であることが判明したため、ノンフロン系の冷媒が使用されるようになってきた。電気冷蔵庫の構造は、圧縮式冷凍機、貯蔵室、運転制御装置からなっている。圧縮冷凍機は冷媒ガスを圧縮し、そのガスを凝縮器で液化させ、その液化冷媒を毛細管を通すことによって圧力を下げ、冷却器で蒸発させるという機能を連続的に行う。貯蔵室は食物を入れる容器で、冷凍機により冷却された貯蔵室内の温度が外気によって上昇しないように、内箱と外箱の間に断熱材を入れてある。運転制御装置は、貯蔵室内の温度を適当に保つための装置で、温度を感知するセンサーが各貯蔵室に設けられており、冷蔵室は3℃前後、冷凍室は零下18℃前後に調節できるようになっている。冷蔵庫の使用時には、扉の開閉中に侵入した外気中の水蒸気および貯蔵室内の食品から蒸発した水蒸気が冷却器に霜となって付着するために、貯蔵室内の冷却状態が悪くなる。この霜を取り除く装置が霜取り装置で、自動的に行うもの、ダイヤルまたは押しボタンなどによる手動式のものとがある。冷蔵庫内の冷却の方式には、冷気自然対流式とファンを使った冷気強制循環式がある。冷凍冷蔵庫の形態は、冷凍室と冷蔵室のそれぞれに扉のあるツードア型が一般的であるが、冷凍冷蔵庫の大型化への対応および使用上での便利性を考慮したスリードア、あるいはそれ以上の扉のあるものも登場している。冷凍冷蔵庫の冷凍室の性能は、JISに定められていて、平均冷凍負荷の温度によって規定されている。すなわち、冷蔵室の温度が0℃以下にならない前提で、冷凍室の負荷の温度が零下18℃以下のもの、零下15℃以下のもの、および零下12℃以下のものの3区分に分かれていて、それぞれ記号で表す。記号の呼び方は、「スリースター」「ハイツースター」「ツースター」という。そのほか、冷凍室の容量100リットル当り4.5キログラム以上の食品を、24時間以内に零下18℃以下に凍結できる冷凍室を「フォースター」という。冷凍食品の保存期間は、食品の種類、冷凍方法、冷凍するまでの食品の履歴などにより一概にいえないが、フォースターとスリースターは約3か月、ハイツースターは約1・8か月、ツースターは約1か月が目安とされている。電気冷蔵庫を据え付ける場所は、直射日光の当たらないところ、ガス台など発熱器具から離れたところ、水がかからない湿気の少ないところを選ぶ。また、凝縮器の放熱を妨げないように注意することもたいせつである。冷蔵庫内の食品の温度上昇を防ぎ、電気エネルギーを節約するためには、食品を詰めすぎたり、熱い食品をそのまま入れたりしないようにする。また、むだな扉の開閉や扉の閉め忘れなどに注意することも必要である。"
# , "中型、小型の電気冷蔵庫。1913年にアメリカで実用化され、20年代に日本に輸入された。国産第1号の電気冷蔵庫は、1930年（昭和5）に発売されたが、価格は700円以上で家が一軒買えるほど高価なものであった。JIS(ジス)（日本工業規格）によれば、電気によって駆動される圧縮冷凍機によって冷却され、冷凍食品以外の食品を貯蔵するために必要な温度を保つことができる貯蔵室（冷蔵室という）が一つ以上あるものを冷蔵庫といい、冷蔵室を一つ以上もち、なおかつ冷凍食品を貯蔵するために必要な温度を保つことができる貯蔵室（冷凍室という）が一つ以上あるものを冷凍冷蔵庫といい、これらを総称したものが電気冷蔵庫である、ということになる。電気冷蔵庫の原理は、液体が気化する際の気化熱を利用したもので、これに使用する液体（冷媒）は、一般にフロンガス（ジクロロジフルオロメタン）が使用されていたが、オゾン層破壊が問題となり、使用禁止となったため、代替フロンのHFC（ハイドロフルオロカーボン）などが用いられている。この代替フロンもきわめて高い温室効果をもたらす物質であることが判明したため、ノンフロン系の冷媒が使用されるようになってきた。電気冷蔵庫の構造は、圧縮式冷凍機、貯蔵室、運転制御装置からなっている。圧縮冷凍機は冷媒ガスを圧縮し、そのガスを凝縮器で液化させ、その液化冷媒を毛細管を通すことによって圧力を下げ、冷却器で蒸発させるという機能を連続的に行う。貯蔵室は食物を入れる容器で、冷凍機により冷却された貯蔵室内の温度が外気によって上昇しないように、内箱と外箱の間に断熱材を入れてある。運転制御装置は、貯蔵室内の温度を適当に保つための装置で、温度を感知するセンサーが各貯蔵室に設けられており、冷蔵室は3℃前後、冷凍室は零下18℃前後に調節できるようになっている。冷蔵庫の使用時には、扉の開閉中に侵入した外気中の水蒸気および貯蔵室内の食品から蒸発した水蒸気が冷却器に霜となって付着するために、貯蔵室内の冷却状態が悪くなる。この霜を取り除く装置が霜取り装置で、自動的に行うもの、ダイヤルまたは押しボタンなどによる手動式のものとがある。冷蔵庫内の冷却の方式には、冷気自然対流式とファンを使った冷気強制循環式がある。冷凍冷蔵庫の形態は、冷凍室と冷蔵室のそれぞれに扉のあるツードア型が一般的であるが、冷凍冷蔵庫の大型化への対応および使用上での便利性を考慮したスリードア、あるいはそれ以上の扉のあるものも登場している。冷凍冷蔵庫の冷凍室の性能は、JISに定められていて、平均冷凍負荷の温度によって規定されている。すなわち、冷蔵室の温度が0℃以下にならない前提で、冷凍室の負荷の温度が零下18℃以下のもの、零下15℃以下のもの、および零下12℃以下のものの3区分に分かれていて、それぞれ記号で表す。記号の呼び方は、「スリースター」「ハイツースター」「ツースター」という。そのほか、冷凍室の容量100リットル当り4.5キログラム以上の食品を、24時間以内に零下18℃以下に凍結できる冷凍室を「フォースター」という。冷凍食品の保存期間は、食品の種類、冷凍方法、冷凍するまでの食品の履歴などにより一概にいえないが、フォースターとスリースターは約3か月、ハイツースターは約1・8か月、ツースターは約1か月が目安とされている。電気冷蔵庫を据え付ける場所は、直射日光の当たらないところ、ガス台など発熱器具から離れたところ、水がかからない湿気の少ないところを選ぶ。また、凝縮器の放熱を妨げないように注意することもたいせつである。冷蔵庫内の食品の温度上昇を防ぎ、電気エネルギーを節約するためには、食品を詰めすぎたり、熱い食品をそのまま入れたりしないようにする。また、むだな扉の開閉や扉の閉め忘れなどに注意することも必要である。"
# , "マイクロ波の性質を利用して食品を加熱する調理器具。1955年アメリカのレイセオン社で初めて商品化され、日本では61年（昭和36）東京芝浦電気（現東芝）で国産1号機がつくられた。マイクロ波は電磁波の一種で、ガラス、紙などを透過し、金属によって反射されるが、食品、水などには吸収されやすい性質をもっている。吸収された電磁波エネルギーは熱に変わり、その物質を発熱させる。電子レンジで使われている電波は、国際的に割り当てられた2450メガヘルツという高い周波数で、マグネトロンで発生させる。マグネトロンは従来レーダーや通信に用いられていた超高周波用の真空管で、加えられた高圧直流電力の60％以上をマイクロ波電力に変換できて効率が高い。マグネトロンで発生させたマイクロ波は、食品にできるだけ均一に当て、加熱むらが生じないように金属製の攪拌(かくはん)翼を設けるか、食品をのせ加熱中回転させるようにくふうされている。一般に食品はマイクロ波の吸収が多く、内部で熱に変換されるが、この現象に寄与しているのは、食品の中に60～96％含まれている水である。電子レンジは、加熱品目に応じてその出力を可変できるものがあり、弱い出力は卵料理やなま物の解凍に使われている。人間が考え、行ってきた調理法は、食品の外部から熱を加え、熱伝導によって内部まで加熱調理する方法であった。ところが食品は一般に熱伝導が低いために、短時間で中心まで急速に加熱して調理しようとすれば、表面と中心との温度勾配(こうばい)が大きくなる。そのため、表面は過熱して組織が破壊し、焦げなどを生じて栄養分が失われる結果となる。これに反し、マイクロ波による加熱調理は、ガラス、陶磁器、プラスチックなどの電波を通しやすい容器を使って加熱すると、食品自体が内部から加熱するので、調理品目によってはほかの加熱方法より効率が高く、調理時間が短縮され、ビタミンの残存率も高くなる。また電波を通しやすい材料の容器やシートで包んだまま加熱する方法、冷凍食品の解凍、温め加熱、殺菌効果の利用なども、電子レンジの特徴を生かした利用方法といえる。"
# , "食品を蒸し焼きにするための調理器具。天火ともいう。一般に箱形で、熱せられた空気と食品から発生する水蒸気を器内に閉じ込め、100℃以上に加熱して調理を行う。したがって直火(じかび)焼きのように食品が強い熱で焦げたり、強い収縮で固くなったりすることもなく、口あたりや風味のよい料理がつくりやすい。オーブンの原理は非常に古く、1万年以上前から知られていたようで、古くは土に掘った穴の中で十分に火を燃やしたあと、動物などの獲物を入れ、その上から焼いた小石や土をかけて蒸し焼きを行っていた。のちに土でつくったかまど形オーブンやれんが積みオーブンへと発展するが、金属が使えるようになって、現在のような小形のオーブンがつくられるようになった。すでにローマ時代にパンが焼かれていることから、この時代にかまど形オーブンが用いられていたことが想像できる。"
# , "コーヒーをいれる道具。サイホン・パーコレーターなど各種についていうが、特に電動のドリップ式のものを指す。"
# , "電力を利用して衣類を洗う機械。電気洗濯機ともいう。普通洗濯は「洗い」「すすぎ」「脱水」「乾燥」の四つの工程に分けられるが、洗濯機は、そのうち脱水までの三つの工程を行うものが多い。乾燥まで行う洗濯乾燥機もある。"
# , "電気やガスを熱源として洗濯物などを乾かす機器。衣類乾燥機はドラムを回転させて遠心力で衣類の水分を飛ばし、加熱した空気をドラム内に送って乾燥させる。洗濯機と一体になった洗濯乾燥機もある。そのほか布団乾燥機などがある。"
# , "電熱線を利用した乾燥機。熱源から水分が出ないため，繊維製品などの乾燥に適している。他の熱源を使う乾燥機に比べて温度調整が簡単にできる"]

content = ["自動車、車"
, "ピアノ、打弦楽器、楽器"
, "電気機器。機械。テレビまたは専用カメラを通して送られてくる画像・音声を、磁気テープに記録したり、それを再生したりする装置"
, "室内の空気調和を行う装置。機械。家電。単にエアコンともいう"
, "機械。コンピュータの分類の一つを表し、個人所有のコンピュータをさすが、明確な定義はなく、オフィスで使用するコンピュータをさす場合もある。日本では縮めてパソコンという。"
, "機械。文書を効率よく作成するための機器またはプログラム。"
, "デジタルオーディオディスクの一つ。デジタル化した音声を円盤に記録したレコード。略してCDともいう。"
, "ビデオ信号すなわち映像と音の信号を記録したレコード。"
, "2輪の自動車の総称。車"
, "2輪の車。車"
, "中型、小型の電気冷蔵庫。家電。食品を保存する"
, "大型の電気冷蔵庫。家電。食品を保存する"
, "マイクロ波の性質を利用して食品を加熱する調理器具。機械。家電。料理。"
, "食品を蒸し焼きにするための調理器具。機械。家電。料理。"
, "コーヒーをいれる道具。"
, "電力を利用して衣類を洗う機械。家電。"
, "ガスを熱源として洗濯物などを乾かす機械。"
, "電熱線を利用して洗濯物などを乾かす機械。"]

content_name = ["自動車", "ピアノ", "VTR", "エアコン", "PC", "ワープロ", "CD", "VD", "自動二輪車", "自転車", "大型電気冷蔵庫", "中、小型電気冷蔵庫", "電子レンジ", "オーブン", "コーヒーメーカー", "電気洗濯機", "衣料乾燥機", "電気乾燥機"]

words = []
content_words = [[] for i in range(len(content))]
t = mc.Tagger('-Ochasen')
t.parse("")

# 単語のリスト words
# 商品ごとの単語のリスト content_words
for i in range(len(content_words)):
    node = t.parseToNode(content[i])
    while node:
        w = node.feature.split(',')
        if out_symbol(w[6]):
            words.append(w[6])
            content_words[i].append(w[6])
        node = node.next

examples = pd.Series(words).value_counts()
print(examples)
vocabulary = pd.Series(words).unique()
words_size = len(words)
vocabulary_size = len(vocabulary)

# ユーザーが評価したコンテンツの数
n_user_rated = np.zeros(user_n)
for user in range(user_n):
    for item in range(item_n):
        if train_data_sparse[user][item] != -1:
            n_user_rated[user] += 1

# 評価した単語の数(種類数)
words_user_rated = [[] for i in range(user_n)]

for user in range(user_n):
    # 評価したコンテンツの単語を入れていく
    for item in range(item_n):
        if train_data_sparse[user][item] != -1:
            words_user_rated[user].append(content_words[item])
    words_user_rated[user] = list(itertools.chain.from_iterable(words_user_rated[user]))
    words_user_rated[user] = pd.Series(words_user_rated[user])
    words_user_rated[user] = words_user_rated[user].unique()
    words_user_rated[user] = len(words_user_rated[user])

# 評価した単語の総数(重複あり)
tmp = np.zeros(user_n)
tmp_w = np.zeros(user_n)
for i in range(user_n):
    for j in range(item_n):
        if train_data_sparse[i][j] != -1:
            tmp[i] += len(content_words[j])

p_c = np.zeros((user_n, 2))
full_train_data = deepcopy(train_data_sparse)
for user in tqdm(range(user_n)):
    text = [[] for i in range(2)] # text[0]には、userが評価0とした単語
    docs = [[] for i in range(2)]
    n_k = [[] for i in range(2)]
    #p_cの計算
    for c in range(2): # 評価が01なので、2回ループ
        for item in range(item_n):
            if train_data_sparse[user][item] == c:
                text[c].append(content_words[item]) # userがcと評価した単語をまとめる
                # p_c[user][c] += len(content_words[item])
                p_c[user][c] += 1 # userがcと評価した数
        # p_c[user][c] = (p_c[user][c] * words_size + 1) / (tmp[user] * words_size + 2)
        p_c[user][c] = (p_c[user][c] * words_size + 1) / (n_user_rated[user] * words_size + 2)
        text[c] = list(itertools.chain.from_iterable(text[c]))
        text[c] = pd.Series(text[c])
        docs[c] = text[c].unique()
        n_k[c] = text[c].value_counts(sort = False)

    p_asc = np.zeros((2, vocabulary_size))
    for c in range(2):
        for i in range(len(docs[c])):
            p_asc[c][vocabulary.tolist().index(docs[c][i])] = (n_k[c][docs[c][i]] * words_size + 1) / (words_size * len(text[c]) + words_user_rated[user])
            # p_asc[c][vocabulary.tolist().index(docs[c][i])] *= idf(docs[c][i], content_words)

    for c in range(2):
        for i in range(vocabulary_size):
            if p_asc[c][i] == 0:
                p_asc[c][i] = 1.0 / (words_size * len(text[c]) + words_user_rated[user])
                # p_asc[c][i] *= idf(vocabulary[i], content_words)

    for item in range(item_n):
        if full_train_data[user][item] == -1:
            p = [[] for i in range(2)]
            for c in range(2):
                p[c] = math.log10(p_c[user][c])
                for i in range(len(content_words[item])):
                    p[c] += math.log10(p_asc[c][vocabulary.tolist().index(content_words[item][i])])
            full_train_data[user][item] = np.argmax(p)

cb_prediction_matrix = deepcopy(full_train_data)
for i in tqdm(range(user_n)):
    for j in range(item_n):
        if train_data_sparse[i][j] != -1:
            cb_prediction_matrix[i][j] = 0

cb_prediction = np.zeros(len(correct_value))
for i in range(len(correct_value)):
    cb_prediction[i] = full_train_data[int(test_data[i][0])][int(test_data[i][1])]

sw_max = 0.2

self_weight = (n_user_rated / 50) * sw_max
for i in range(user_n):
    if self_weight[i] > sw_max:
        self_weight[i] = sw_max

#ユーザーごとの、ratingの偏差を求める
rating_average = np.average(full_train_data, axis=1)
rate_deviation = deepcopy(full_train_data)
for i in range(item_n):
    rate_deviation[:,i] -= rating_average

#ユーザーの相関係数を求める
rate_deviation_sum = np.sum(rate_deviation, axis=1)
user_similarity = np.zeros((user_n, user_n))
for i in range(user_n):
    for j in range(user_n):
        if i != j:
            user_similarity[i][j] = cos_sim(full_train_data[i], full_train_data[j])
# print(user_similarity)
# print(user_similarity[0][773])

# user_similarity = deepcopy(full_train_data)
# user_similarity = np.corrcoef(user_similarity)
# print(user_similarity)

# for i in range(user_n):
#     user_similarity[i][i] = 0

#上位30のユーザーをneighborhoodとする
neighborhood = np.argsort(user_similarity, axis=1)[:, ::-1][:, 1:31]
# print(neighborhood)

#neighborのhm
m = np.zeros(user_n)
for i in range(user_n):
    if n_user_rated[i] >= 50:
        m[i] = 1
    elif n_user_rated[i] < 50:
        m[i] = n_user_rated[i] / 50

hm = np.zeros((user_n, neighborhood.shape[1]))
for i in range(user_n):
    for j in range(neighborhood.shape[1]):
        hm[i][j] = 2 * m[i] * m[neighborhood[i][j]] / (m[i] + m[neighborhood[i][j]])

#neighborのsg
sg = np.zeros((user_n, neighborhood.shape[1]))

co_rated_item = np.zeros((user_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(neighborhood.shape[1]):
        for k in range(item_n):
            if train_data_sparse[i][k] != -1 and train_data_sparse[neighborhood[i][j]][k] != -1:
                co_rated_item[i][j] += 1
        if co_rated_item[i][j] >= 50:
            co_rated_item[i][j] = 50
sg = co_rated_item / 50

#neighborのhw
hw = hm + sg

#neighborの偏差と相関係数
corr_neighbor = np.zeros((user_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(neighborhood.shape[1]):
        corr_neighbor[i][j] = user_similarity[i][neighborhood[i][j]]

corr_deviation = np.zeros((user_n, item_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(item_n):
        for k in range(neighborhood.shape[1]):
            corr_deviation[i][j][k] = rate_deviation[neighborhood[i][k]][j]

#hw * p * (v-v)
hwpvv = deepcopy(corr_deviation)
for i in range(user_n):
    for j in range(item_n):
        hwpvv[i][j] *= corr_neighbor[i] * hw[i]
hwpvv_sum = np.zeros((user_n, item_n))

for i in range(user_n):
    tmp_4 = np.sum(hwpvv[i], axis=1)
    for j in range(item_n):
        hwpvv_sum[i][j] = tmp_4[j]

hwp = hw * corr_neighbor
hwp_sum = np.sum(hwp, axis=1)

#prediction
cbcf_prediction_matrix = np.zeros((user_n, item_n))
for i in tqdm(range(user_n)):
    for j in range(item_n):
        a = self_weight[i] * (full_train_data[i][j] - rating_average[i]) + hwpvv_sum[i][j]
        b = self_weight[i] + hwp_sum[i]
        cbcf_prediction_matrix[i][j] = rating_average[i] + (a / b)
        # cbcf_prediction_matrix[i][j] = rating_average[i] + (np.sum(corr_deviation[i][j] * corr_neighbor[i]) / np.sum(corr_neighbor[i]))
for i in tqdm(range(user_n)):
    for j in range(item_n):
        if train_data_sparse[i][j] != -1:
            cbcf_prediction_matrix[i][j] = 0

cbcf_prediction = np.zeros(len(correct_value))
for i in range(len(correct_value)):
    cbcf_prediction[i] = cbcf_prediction_matrix[int(test_data[i][0])][int(test_data[i][1])]

#MAE
tmp_7 = np.abs(correct_value - cbcf_prediction)
cbcf_mae = np.sum(tmp_7) / len(correct_value)
tmp_7 = np.abs(correct_value - cb_prediction)
cb_mae = np.sum(tmp_7) / len(correct_value)

# cf

#ユーザーごとの、ratingの偏差を求める
rating_average = np.sum(train_data, axis=1) / n_user_rated
rate_deviation = deepcopy(train_data)
for i in range(item_n):
    rate_deviation[:,i] -= rating_average

for i in range(user_n):
    for j in range(item_n):
        if train_data_sparse[i][j] == -1:
            rate_deviation[i][j] = 0


#ユーザーの相関係数を求める
rate_deviation_sum = np.sum(rate_deviation, axis=1)
user_similarity = np.zeros((user_n, user_n))
for i in range(user_n):
    for j in range(user_n):
        if i != j:
            user_similarity[i][j] = cos_sim(train_data_sparse[i], train_data_sparse[j])

#上位30のユーザーをneighborhoodとする
neighborhood = np.argsort(user_similarity, axis=1)[:, ::-1][:, 1:31]

#neighborのhm
m = np.zeros(user_n)
for i in range(user_n):
    if n_user_rated[i] >= 50:
        m[i] = 1
    elif n_user_rated[i] < 50:
        m[i] = n_user_rated[i] / 50

hm = np.zeros((user_n, neighborhood.shape[1]))
for i in range(user_n):
    for j in range(neighborhood.shape[1]):
        hm[i][j] = 2 * m[i] * m[neighborhood[i][j]] / (m[i] + m[neighborhood[i][j]])

#neighborのsg
sg = np.zeros((user_n, neighborhood.shape[1]))

co_rated_item = np.zeros((user_n, neighborhood.shape[1]))
for i in range(user_n):
    for j in range(neighborhood.shape[1]):
        for k in range(item_n):
            if train_data_sparse[i][k] != -1 and train_data_sparse[neighborhood[i][j]][k] != -1:
                co_rated_item[i][j] += 1
        if co_rated_item[i][j] >= 50:
            co_rated_item[i][j] = 50
sg = co_rated_item / 50

#neighborのhw
hw = hm + sg

#neighborの偏差と相関係数
corr_neighbor = np.zeros((user_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(neighborhood.shape[1]):
        corr_neighbor[i][j] = user_similarity[i][neighborhood[i][j]]

corr_deviation = np.zeros((user_n, item_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(item_n):
        for k in range(neighborhood.shape[1]):
            corr_deviation[i][j][k] = rate_deviation[neighborhood[i][k]][j]

#hw * p * (v-v)
hwpvv = deepcopy(corr_deviation)
for i in range(user_n):
    for j in range(item_n):
        hwpvv[i][j] *= corr_neighbor[i] * hw[i]
hwpvv_sum = np.zeros((user_n, item_n))

for i in range(user_n):
    tmp_4 = np.sum(hwpvv[i], axis=1)
    for j in range(item_n):
        hwpvv_sum[i][j] = tmp_4[j]

hwp = hw * corr_neighbor
hwp_sum = np.sum(hwp, axis=1)

#prediction
cf_prediction_matrix = deepcopy(train_data_sparse)
for i in range(user_n):
    for j in range(item_n):
        if train_data_sparse[i][j] == -1:
            a = hwpvv_sum[i][j]
            b = hwp_sum[i]
            cf_prediction_matrix[i][j] = rating_average[i] + (a / b)
            # cf_prediction_matrix[i][j] = rating_average[i] + (np.sum(corr_deviation[i][j] * corr_neighbor[i]) / np.sum(corr_neighbor[i]))
for i in range(user_n):
    for j in range(item_n):
        if train_data_sparse[i][j] != -1:
            cf_prediction_matrix[i][j] = 0

cf_prediction = np.zeros(len(correct_value))
for i in range(len(correct_value)):
    cf_prediction[i] = cf_prediction_matrix[int(test_data[i][0])][int(test_data[i][1])]

#MAE
tmp_7 = np.abs(correct_value - cf_prediction)
cf_mae = np.sum(tmp_7) / len(correct_value)
print("cf", cf_mae)
print("cb", cb_mae, "cbcf", cbcf_mae)
# print(cf_prediction, cf_prediction.shape)
# print(cb_prediction, cb_prediction.shape)
# print(cbcf_prediction, cbcf_prediction.shape)
# print(correct_value, correct_value.shape)

cf_reclist = np.argsort(cf_prediction_matrix, axis=1)[:, ::-1][:, 1:31]
cb_reclist = np.argsort(full_train_data, axis=1)[:, ::-1][:, 1:31]
cbcf_reclist = np.argsort(cbcf_prediction_matrix, axis=1)[:, ::-1][:, 1:31]
print("cf", cf_reclist)
print("cb", cb_reclist)
print("cbcf", cbcf_reclist)

content_name_csv = []
content_name_miss = np.zeros(item_n)
for i in range(len(correct_value)):
    content_name_csv.append(content_name[int(test_data[i][1])])
    if cb_prediction[i] != correct_value[i]:
        content_name_miss[int(test_data[i][1])] += 1

with open('nonidf_need_scan_cb.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(content_name_csv)
    writer.writerow(cb_prediction)
    writer.writerow(correct_value)
    writer.writerow(content_name)
    writer.writerow(content_name_miss)

# FPR, TPR(, しきい値) を算出
fpr, tpr, thresholds = metrics.roc_curve(correct_value, cf_prediction)

# ついでにAUCも
auc = metrics.auc(fpr, tpr)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='CF (AUC = %.2f)'%auc)
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# FPR, TPR(, しきい値) を算出
fpr, tpr, thresholds = metrics.roc_curve(correct_value, cbcf_prediction)

# ついでにAUCも
auc = metrics.auc(fpr, tpr)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='CBCF (AUC = %.2f)'%auc)
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)


k=3

pred = KMeans(n_clusters=k).fit_predict(train_data)

c = [[] for i in range(k)]
recommend_rate = [[] for i in range(k)]
for j in range(k):
    for i in range(996):
        if pred[i] == j:
            c[j].append(train_data[i])
    recommend_rate[j] = np.sum(c[j], axis=0) / len(c[j])

for i in range(996):
    train_data[i] = recommend_rate[pred[i]]

k_prediction = np.zeros(len(correct_value))
for i in range(len(correct_value)):
    k_prediction[i] = train_data[int(test_data[i][0])][int(test_data[i][1])]

# k_prediction = np.round(k_prediction)
tmp_7 = np.abs(correct_value - k_prediction)
k_mae = np.sum(tmp_7) / len(correct_value)

print(k_mae)

# FPR, TPR(, しきい値) を算出
fpr, tpr, thresholds = metrics.roc_curve(correct_value, k_prediction)

# ついでにAUCも
auc = metrics.auc(fpr, tpr)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='kmeans (AUC = %.2f)'%auc)
plt.legend()
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()
plt.savefig('nonidf_need_scan_cbcf_roc_curve.png')