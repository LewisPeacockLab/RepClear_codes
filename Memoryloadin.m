clear all

data_path=('/Users/zb3663/Dropbox (LewPeaLab)/BEHAVIOR/repclear/v2_pilot');
cd(data_path)
foldernames=dir('repclear*');

for a=1:17%length({foldernames.name})
cd([data_path,'/',foldernames(a).name]);
memnames(a)=dir('*_memory_*.mat');
designnames(a)=dir('*design*.mat');
datastruc(1).(foldernames(a).name)=load(memnames(a).name);
cd ..
end


%This above script loads in the memory files
%%
%AUC%
temp_matrix=datastruc.(foldernames(1).name).xdata.matrix;
temp_header=datastruc.(foldernames(1).name).args.design.ph{4}.header;
temp_correct=temp_matrix(:, findCol(temp_header, {'accuracy'}));
temp_response=temp_matrix(:, findCol(temp_header, {'response'}));

one_resp=temp_correct(find(temp_response==1));
two_resp=temp_correct(find(temp_response==2));
three_resp=temp_correct(find(temp_response==3));
four_resp=temp_correct(find(temp_response==4));

p1c=sum(one_resp)/length(temp_correct);
p1w=(length(one_resp)-sum(one_resp))/length(temp_correct);

p2c=sum(two_resp)/length(temp_correct);
p2w=(length(two_resp)-sum(two_resp))/length(temp_correct);

p3c=sum(three_resp)/length(temp_correct);
p3w=(length(three_resp)-sum(three_resp))/length(temp_correct);

p4c=sum(four_resp)/length(temp_correct);
p4w=(length(four_resp)-sum(four_resp))/length(temp_correct);

Ptable=[p1c,p1w;(p1c+p2c),(p1w+p2w);(p1c+p2c+p3c),(p1w+p2w+p3w);(p1c+p2c+p3c+p4c),(p1w+p2w+p3w+p4w)];

temp_Ptable=Ptable;
temp_Ptable=[0,0;temp_Ptable;1,1];
temp_Pfit=fit(temp_Ptable(:,2),temp_Ptable(:,1),'poly');
figure;plot(temp_Pfit,temp_Ptable(:,2),temp_Ptable(:,1));

Pfit=fit(Ptable(:,2),Ptable(:,1),'exp1');
figure;scatter(Ptable(:,2),Ptable(:,1)); hold on;plot([0,1],[0,1]);

%%
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


