clear all

data_path=('/Users/zb3663/Dropbox (LewPeaLab)/BEHAVIOR/repclear/v2_pilot');
cd(data_path)
foldernames=dir('repclear*');

for a=1:length({foldernames.name})
cd([data_path,'/',foldernames(a).name]);
names=dir('*_memory_*.mat');
datastruc(1).(foldernames(a).name)=load(names.name);
cd ..
end

temp_matrix=xdata.matrix;
temp_header=args.design.ph{4}.header;

temp_rts=temp_matrix(:, findCol(temp_header, {'rt'}));
temp_correct=temp_matrix(:, findCol(temp_header, {'accuracy'}));
temp_response=temp_matrix(:, findCol(temp_header, {'response'}));
temp_old_new=temp_matrix(:, findCol(temp_header, {'old_lure_novel'}));
temp_pre_post=temp_matrix(:, findCol(temp_header, {'pre_post'}));

temp_correct_single=temp_correct(find(temp_pre_post==1));
temp_correct_double=temp_correct(find(temp_pre_post==2));
temp_correct_zero=temp_correct(find(temp_pre_post==0));


temp_correct_rts=temp_rts(find(temp_correct==1 & (temp_response== 1 | temp_response==4)));
mean_tcrts=mean(temp_correct_rts);
temp_incorrect_rts=temp_rts(find(temp_correct==0));
mean_tirts=mean(temp_incorrect_rts);
std_tirts=std(temp_incorrect_rts);
std_tcrts=std(temp_correct_rts);

figure;errorbar([mean_tcrts,mean_tirts],[std_tcrts,std_tirts]);


