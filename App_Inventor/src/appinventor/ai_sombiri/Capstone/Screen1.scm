#|
$JSON
{"authURL":["ai2.appinventor.mit.edu"],"YaVersion":"167","Source":"Form","Properties":{"$Name":"Screen1","$Type":"Form","$Version":"23","AlignHorizontal":"3","AlignVertical":"2","AppName":"GUI_Mockup","Scrollable":"True","Title":"Welcome","TitleVisible":"False","Uuid":"0","$Components":[{"$Name":"login_arr","$Type":"VerticalArrangement","$Version":"3","AlignHorizontal":"3","Uuid":"335812916","Visible":"False","$Components":[{"$Name":"logo_img","$Type":"Image","$Version":"3","Picture":"157611767e2d8e8c0205dc273dbc2d51cf6e9e29b4be2e3d-stocklarge.jpg","ScalePictureToFit":"True","Uuid":"477466265"},{"$Name":"uname_lbl","$Type":"Label","$Version":"4","Text":"Username","Uuid":"955793344"},{"$Name":"uname_tbox","$Type":"TextBox","$Version":"5","Text":"Sombiri","Uuid":"209320707"},{"$Name":"login_lbl","$Type":"Label","$Version":"4","Text":"Login","Uuid":"-466964125"},{"$Name":"pword_tbox","$Type":"PasswordTextBox","$Version":"4","Uuid":"-580592632"},{"$Name":"login_ha","$Type":"HorizontalArrangement","$Version":"3","Uuid":"-1173499999","$Components":[{"$Name":"go_btn","$Type":"Button","$Version":"6","Text":"Go","Uuid":"1639014635"},{"$Name":"reg_btn","$Type":"Button","$Version":"6","Text":"Register","Uuid":"2103045085"}]},{"$Name":"DEBUG_LBL","$Type":"Label","$Version":"4","Uuid":"1928524876"}]},{"$Name":"new_user_arr","$Type":"VerticalArrangement","$Version":"3","AlignHorizontal":"3","Uuid":"1897605990","Visible":"False","$Components":[{"$Name":"uname_lbl2","$Type":"Label","$Version":"4","Text":"Username","Uuid":"292293075"},{"$Name":"uname_tbox2","$Type":"TextBox","$Version":"5","Hint":"eg btarth7","Uuid":"-1346844618"},{"$Name":"uname_unique_lbl","$Type":"Label","$Version":"4","Uuid":"1059222465"},{"$Name":"passwd_lbl","$Type":"Label","$Version":"4","Text":"Password","Uuid":"-1074842536"},{"$Name":"passwd_tbox","$Type":"PasswordTextBox","$Version":"4","Uuid":"368645124"},{"$Name":"passwd_conf_lbl0","$Type":"Label","$Version":"4","Text":"Password Confirmation","Uuid":"157796250"},{"$Name":"passwd_conf_lbl1","$Type":"Label","$Version":"4","Uuid":"327638527"},{"$Name":"passwd_conf_tbox","$Type":"PasswordTextBox","$Version":"4","Uuid":"1723420109"},{"$Name":"fname_lbl","$Type":"Label","$Version":"4","Text":"Full Name","Uuid":"-826804380"},{"$Name":"fname_tbox","$Type":"TextBox","$Version":"5","Hint":"eg. Brianne Tarth","Uuid":"-1557528184"},{"$Name":"age_lbl","$Type":"Label","$Version":"4","Text":"Age","Uuid":"-1063443993"},{"$Name":"age_tbox","$Type":"TextBox","$Version":"5","Hint":"in years","NumbersOnly":"True","Uuid":"-1479385512"},{"$Name":"gender_lbl","$Type":"Label","$Version":"4","Text":"Birth Sex","Uuid":"-1364940624"},{"$Name":"gender_spnr","$Type":"Spinner","$Version":"1","ElementsFromString":"Female, Male","Uuid":"247706474"},{"$Name":"race_lbl","$Type":"Label","$Version":"4","Text":"Race","Uuid":"2100214126"},{"$Name":"race_spnr","$Type":"Spinner","$Version":"1","ElementsFromString":"American Indian, Asian, Black, Pacific Islander, White, mixed_other ","Uuid":"-1165310716"},{"$Name":"skn_typ_lbl","$Type":"Label","$Version":"4","Text":"Skin Type","Uuid":"243682183"},{"$Name":"skn_typ_spnr","$Type":"Spinner","$Version":"1","ElementsFromString":"normal, oily, dry","Uuid":"2123276806"},{"$Name":"acne_cbox","$Type":"CheckBox","$Version":"2","Text":"Do you have acne now?","Uuid":"665716447"},{"$Name":"irr_prod_lbl","$Type":"Label","$Version":"4","Text":"Add products which have caused irritation","Uuid":"2063993788"},{"$Name":"HorizontalArrangement2","$Type":"HorizontalArrangement","$Version":"3","Width":"-2","Uuid":"-1369979640","$Components":[{"$Name":"irr_prod_name_tbox","$Type":"TextBox","$Version":"5","Width":"-2","Hint":"Stop typing for autocompletion","Uuid":"982719007"},{"$Name":"irr_prod_add_btn","$Type":"Button","$Version":"6","Text":"Add","Uuid":"-2097462237"}]},{"$Name":"result_lbl","$Type":"Label","$Version":"4","Uuid":"-488269770"},{"$Name":"get_strtd_btn","$Type":"Button","$Version":"6","Text":"Get Started","Uuid":"450036172"}]},{"$Name":"barcode_arr","$Type":"VerticalArrangement","$Version":"3","AlignHorizontal":"3","Width":"-2","Uuid":"1859390536","Visible":"False","$Components":[{"$Name":"actionBar_ha_barcode","$Type":"HorizontalArrangement","$Version":"3","Width":"-2","Uuid":"-651374100"},{"$Name":"Scan_Barcode","$Type":"Button","$Version":"6","Text":"scan bar code","Uuid":"-1346053603"},{"$Name":"ha_barcode1","$Type":"HorizontalArrangement","$Version":"3","Uuid":"-820365941","$Components":[{"$Name":"ReturnBarcode","$Type":"TextBox","$Version":"5","Hint":"Hint for TextBox1","Uuid":"-1337933201"},{"$Name":"GotoUrl","$Type":"Button","$Version":"6","Text":"Go to Site","Uuid":"-1823755938"}]},{"$Name":"barcode_TextBox1","$Type":"TextBox","$Version":"5","Hint":"Hint for TextBox1","Uuid":"17353165"},{"$Name":"barcode_Label1","$Type":"Label","$Version":"4","Text":"Text for Label1","Uuid":"1480316389"},{"$Name":"BarcodeScanner1","$Type":"BarcodeScanner","$Version":"2","UseExternalScanner":"False","Uuid":"1547706042"}]},{"$Name":"ocr_arr","$Type":"VerticalArrangement","$Version":"3","AlignHorizontal":"3","Width":"-2","Uuid":"2081908726","$Components":[{"$Name":"actionBar_ha_ocr","$Type":"HorizontalArrangement","$Version":"3","Width":"-2","Uuid":"1171189735"},{"$Name":"TakePic","$Type":"Button","$Version":"6","Text":"Take Picture","Uuid":"-1244908337"},{"$Name":"ocr_status_lbl","$Type":"Label","$Version":"4","Text":"Text for Label9","Uuid":"-1490502885"},{"$Name":"post","$Type":"Button","$Version":"6","Text":"post","Uuid":"1902205218"},{"$Name":"Image1","$Type":"Image","$Version":"3","Uuid":"1906379807"},{"$Name":"ocr_Label1","$Type":"Label","$Version":"4","Uuid":"-1504569336"},{"$Name":"Camera1","$Type":"Camera","$Version":"3","Uuid":"-2138643519"},{"$Name":"TaifunFile1","$Type":"TaifunFile","$Version":"8","Uuid":"2081768709"}]},{"$Name":"product_report_arr","$Type":"VerticalArrangement","$Version":"3","AlignHorizontal":"3","Width":"-2","Uuid":"-2032659950","$Components":[{"$Name":"actionBar_ha_pr","$Type":"HorizontalArrangement","$Version":"3","Width":"-2","Uuid":"2113166840"},{"$Name":"HorizontalArrangement7","$Type":"HorizontalArrangement","$Version":"3","Uuid":"-1272339881","$Components":[{"$Name":"pr_TextBox1","$Type":"TextBox","$Version":"5","Hint":"Type or speak","Uuid":"-636185175"},{"$Name":"start_speaking","$Type":"Button","$Version":"6","Height":"46","Width":"46","Image":"microphone_smol.png","Uuid":"1819691439"}]},{"$Name":"suggest_pkr","$Type":"ListPicker","$Version":"9","Text":"Text for ListPicker1","Uuid":"-1420248907","Visible":"False"},{"$Name":"status_lbl","$Type":"Label","$Version":"4","Uuid":"507270414"},{"$Name":"scoresearch","$Type":"Button","$Version":"6","Text":"Search Score","Uuid":"1173166943"},{"$Name":"VerticalArrangement1","$Type":"VerticalArrangement","$Version":"3","AlignHorizontal":"3","Uuid":"622255064","$Components":[{"$Name":"Label8","$Type":"Label","$Version":"4","Text":"Name","Uuid":"1020906052"},{"$Name":"report_name","$Type":"Label","$Version":"4","FontBold":"True","Uuid":"357750358"},{"$Name":"Label7","$Type":"Label","$Version":"4","Text":"Acne prediction:","Uuid":"133329877"},{"$Name":"predict_img","$Type":"Image","$Version":"3","Uuid":"-970970736"},{"$Name":"report_prediction","$Type":"Label","$Version":"4","Uuid":"-2057789926","Visible":"False"},{"$Name":"Label3","$Type":"Label","$Version":"4","Text":"Overall Score: ","Uuid":"-1716717531"},{"$Name":"report_score","$Type":"Label","$Version":"4","FontBold":"True","Uuid":"363128537"},{"$Name":"Label4","$Type":"Label","$Version":"4","Text":"Cancer Score: ","Uuid":"91002113"},{"$Name":"cancer_score","$Type":"Label","$Version":"4","FontBold":"True","Uuid":"974800860"},{"$Name":"Label5","$Type":"Label","$Version":"4","Text":"Reproduction Toxicity Score:","Uuid":"-1435107099"},{"$Name":"reprod_score","$Type":"Label","$Version":"4","FontBold":"True","Uuid":"-1495363387"},{"$Name":"Label6","$Type":"Label","$Version":"4","Text":"Immune Toxicity Score: ","Uuid":"-951759622"},{"$Name":"imm_score","$Type":"Label","$Version":"4","FontBold":"True","Uuid":"1791991563"}]},{"$Name":"SpeechRecognizer1","$Type":"SpeechRecognizer","$Version":"1","Uuid":"-229912109"}]},{"$Name":"prod_suggest_pkr","$Type":"ListPicker","$Version":"9","Text":"Text for ListPicker1","Uuid":"-1214313112","Visible":"False"},{"$Name":"TaifunTextbox1","$Type":"TaifunTextbox","$Version":"5","Uuid":"1889422706"},{"$Name":"Clock1","$Type":"Clock","$Version":"3","Uuid":"-1944091516"},{"$Name":"TinyDB1","$Type":"TinyDB","$Version":"1","Uuid":"-579774235"},{"$Name":"Web1","$Type":"Web","$Version":"4","Uuid":"1922802509"},{"$Name":"TaifunTools1","$Type":"TaifunTools","$Version":"18","Uuid":"1326181615"},{"$Name":"Actionbar1","$Type":"Actionbar","$Version":"1","ColorBackground":"#ff6700","TextTitleAccionBar":"Barcode Scanner","Uuid":"270647683"},{"$Name":"AndroidThemes1","$Type":"AndroidThemes","$Version":"1","Uuid":"414079514"},{"$Name":"Crop1","$Type":"Crop","$Version":"1","Uuid":"-27194775"},{"$Name":"TaifunImage1","$Type":"TaifunImage","$Version":"4","Uuid":"514598229"}]}}
|#