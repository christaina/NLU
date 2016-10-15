import os

def write(outfile,concat_dict,data_dir):
	with open(outfile,"w") as o:
		for fi,lab in concat_dict.iteritems():
			with open(os.path.join(data_dir,fi),"r") as inf:
				for line in inf.readlines():
                                        line = line.strip("\n")
					o.write("%s __label__%s\n"%(line,lab))


if __name__=="__main__":
	data_dir = "./data/aclImdb"
	files_concat = {'train_pos.txt':'pos','train_neg.txt':'neg'}
                
        files_test = {'test_pos.txt':'pos','test_neg.txt':'neg'}

	outfile_te = "./data/all_input_te.txt"
	outfile_tr = "./data/all_input_tr.txt"
        write(outfile_tr,files_concat,data_dir)
        write(outfile_te,files_test,data_dir)
