source_text = "hp probook 640 g2 8gb ultra slim docking carepack [SEP] hp probook 640 g3 i5 7200u 8gb 256gb ssd 14 w10pro 2013 ultraslim docking station negro black friday 2017	"

sequences_source = source_text.split("[SEP]")

left_source = sequences_source[0].strip()
right_source = sequences_source[1].strip()

from transformers import AutoTokenizer





print("a")