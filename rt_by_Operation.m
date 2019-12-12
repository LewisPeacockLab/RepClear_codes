%RT by Operation
clear all
data_path=('/Users/zb3663/Dropbox (LewPeaLab)/BEHAVIOR/repclear/v2_pilot');
cd(data_path)
foldernames=dir('repclear*');

for a=1:length({foldernames.name})
    cd([data_path,'/',foldernames(a).name]);
    memnames(a)=dir('*_memory_*.mat');
    designnames(a)=dir('*design*.mat');
    datastruc(1).(foldernames(a).name)=load(memnames(a).name);
    cd ..
end

for a=1:length({foldernames.name})
    if a==1 || a==2
        resp.(foldernames(a).name).rt=datastruc.(foldernames(a).name).xdata.matrix(:,[5,14,15]);
    else
        resp.(foldernames(a).name).rt=datastruc.(foldernames(a).name).xdata.matrix(:,[5,13,14]);
    end
    
    resp.(foldernames(a).name).maintain_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==1),3);
    resp.(foldernames(a).name).rep_sub_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==2),3);
    resp.(foldernames(a).name).rep_cat_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==3),3);
    resp.(foldernames(a).name).suppress_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==4),3);
    resp.(foldernames(a).name).clear_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==5),3);
    resp.(foldernames(a).name).new_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==0),3);
    resp.(foldernames(a).name).all_rt=resp.(foldernames(a).name).rt(:,3);
    
    resp.(foldernames(a).name).c_maintain_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==1 & resp.(foldernames(a).name).rt(:,2)==1),3);
    resp.(foldernames(a).name).c_rep_sub_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==2 & resp.(foldernames(a).name).rt(:,2)==1),3);
    resp.(foldernames(a).name).c_rep_cat_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==3 & resp.(foldernames(a).name).rt(:,2)==1),3);
    resp.(foldernames(a).name).c_suppress_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==4 & resp.(foldernames(a).name).rt(:,2)==1),3);
    resp.(foldernames(a).name).c_clear_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==5 & resp.(foldernames(a).name).rt(:,2)==1),3);
    resp.(foldernames(a).name).c_new_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,1)==0 & resp.(foldernames(a).name).rt(:,2)==1),3);
    resp.(foldernames(a).name).c_all_rt=resp.(foldernames(a).name).rt(find(resp.(foldernames(a).name).rt(:,2)==1),3);
    
    resp.(foldernames(a).name).mean_maintain_rt=mean(resp.(foldernames(a).name).maintain_rt);
    resp.(foldernames(a).name).mean_rep_sub_rt=mean(resp.(foldernames(a).name).rep_sub_rt);
    resp.(foldernames(a).name).mean_rep_cat_rt=mean(resp.(foldernames(a).name).rep_cat_rt);
    resp.(foldernames(a).name).mean_suppress_rt=mean(resp.(foldernames(a).name).suppress_rt);
    resp.(foldernames(a).name).mean_clear_rt=mean(resp.(foldernames(a).name).clear_rt);
    resp.(foldernames(a).name).mean_new_rt=mean(resp.(foldernames(a).name).new_rt);
    resp.(foldernames(a).name).mean_all_rt=mean(resp.(foldernames(a).name).all_rt);
    
    resp.(foldernames(a).name).c_mean_maintain_rt=mean(resp.(foldernames(a).name).c_maintain_rt);
    resp.(foldernames(a).name).c_mean_rep_sub_rt=mean(resp.(foldernames(a).name).c_rep_sub_rt);
    resp.(foldernames(a).name).c_mean_rep_cat_rt=mean(resp.(foldernames(a).name).c_rep_cat_rt);
    resp.(foldernames(a).name).c_mean_suppress_rt=mean(resp.(foldernames(a).name).c_suppress_rt);
    resp.(foldernames(a).name).c_mean_clear_rt=mean(resp.(foldernames(a).name).c_clear_rt);
    resp.(foldernames(a).name).c_mean_new_rt=mean(resp.(foldernames(a).name).c_new_rt);
    resp.(foldernames(a).name).c_mean_all_rt=mean(resp.(foldernames(a).name).c_all_rt);
    
    
    groupresp.maintain(a)=resp.(foldernames(a).name).mean_maintain_rt;
    groupresp.rep_sub(a)=resp.(foldernames(a).name).mean_rep_sub_rt;
    groupresp.rep_cat(a)=resp.(foldernames(a).name).mean_rep_cat_rt;
    groupresp.suppress(a)=resp.(foldernames(a).name).mean_suppress_rt;
    groupresp.clear(a)=resp.(foldernames(a).name).mean_clear_rt;
    groupresp.new(a)=resp.(foldernames(a).name).mean_new_rt;
    groupresp.all(a)=resp.(foldernames(a).name).mean_all_rt;
    
    groupresp.c_maintain(a)=resp.(foldernames(a).name).c_mean_maintain_rt;
    groupresp.c_rep_sub(a)=resp.(foldernames(a).name).c_mean_rep_sub_rt;
    groupresp.c_rep_cat(a)=resp.(foldernames(a).name).c_mean_rep_cat_rt;
    groupresp.c_suppress(a)=resp.(foldernames(a).name).c_mean_suppress_rt;
    groupresp.c_clear(a)=resp.(foldernames(a).name).c_mean_clear_rt;
    groupresp.c_new(a)=resp.(foldernames(a).name).c_mean_new_rt;
    groupresp.c_all(a)=resp.(foldernames(a).name).c_mean_all_rt;
end

groupresp.operation_table=[groupresp.maintain',groupresp.rep_sub',groupresp.rep_cat',groupresp.suppress',groupresp.clear',groupresp.new'];
groupresp.c_operation_table=[groupresp.c_maintain',groupresp.c_rep_sub',groupresp.c_rep_cat',groupresp.c_suppress',groupresp.c_clear',groupresp.c_new'];

face_color=[[0.741176470588235,0,0.0235294117647059];[0.992156862745098,0.317647058823529,0];[1,0.607843137254902,0.141176470588235];[0,0.584313725490196,0.537254901960784];[0,0.525490196078431,0.768627450980392];[0.5,0.5,0.5]];


x=1:6;
data=mean(groupresp.operation_table);
stderror=std(groupresp.operation_table)/sqrt(length(groupresp.operation_table));
figure;bar(x,data);
hold on;errorbar(x,data,stderror,'Color',[0 0 0 ],'LineStyle','none');
title('RT by Operation');

c_data=mean(groupresp.c_operation_table);
c_stderror=std(groupresp.c_operation_table)/sqrt(length(groupresp.c_operation_table));
figure;
hold on;
for x=1:6
h=bar(x,c_data(x));
set(h,'Facecolor',face_color(x,:));
end
x=1:6;
hold on;errorbar(x,c_data,c_stderror,'Color',[0 0 0 ],'LineStyle','none');
title('RT by Operation - Correct Only');

%%
%linear regression - RT vs Familiarity

for a=1:length({foldernames.name})
    cd([data_path,'/',foldernames(a).name]);
    fam_names(a)=dir('*Matrix*familiarity*.txt');
    familiarity.(foldernames(a).name).table=readtable(fam_names(a).name);
    cd ..
end

for a=1:length({foldernames.name})
    familiarity.(foldernames(a).name).scores=table2array(familiarity.(foldernames(a).name).table(:,[4,6]));
    for z=1:length(familiarity.(foldernames(a).name).scores)
        
        temp=datastruc.(foldernames(a).name).xdata.matrix(find(datastruc.(foldernames(a).name).xdata.matrix(:,10)==familiarity.(foldernames(a).name).scores(z,1)),14);
        
        familiarity.(foldernames(a).name).scores(z,3)=temp;
        clearvars temp
    end
end

for a=1:length({foldernames.name})
   [famil_stat.(foldernames(a).name).p, famil_stat.(foldernames(a).name).stat] = polyfit(familiarity.(foldernames(a).name).scores(:,2),familiarity.(foldernames(a).name).scores(:,3),1);
   famil_stat.(foldernames(a).name).lm = fitlm(familiarity.(foldernames(a).name).scores(:,2),familiarity.(foldernames(a).name).scores(:,3),'linear');
   famil_stat.(foldernames(a).name).f = polyval(famil_stat.(foldernames(a).name).p,familiarity.(foldernames(a).name).scores(:,2));   
   famil_stat.(foldernames(a).name).resid=familiarity.(foldernames(a).name).scores(:,3)-famil_stat.(foldernames(a).name).f;
end

for a=1:length({foldernames.name})
    for z=1:length(familiarity.(foldernames(a).name).scores(:,1))
        if a==1 || a==2
            familiarity.(foldernames(a).name).rt(z,:)=datastruc.(foldernames(a).name).xdata.matrix(find(datastruc.(foldernames(a).name).xdata.matrix(:,10)==familiarity.(foldernames(a).name).scores(z,1)),[5,14,15]);
        else
            familiarity.(foldernames(a).name).rt(z,:)=datastruc.(foldernames(a).name).xdata.matrix(find(datastruc.(foldernames(a).name).xdata.matrix(:,10)==familiarity.(foldernames(a).name).scores(z,1)),[5,13,14]);
            
        end
    end
    familiarity.(foldernames(a).name).rt(:,4)=familiarity.(foldernames(a).name).rt(:,3)-famil_stat.(foldernames(a).name).f;

    familiarity.(foldernames(a).name).maintain_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==1),4);
    familiarity.(foldernames(a).name).rep_sub_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==2),4);
    familiarity.(foldernames(a).name).rep_cat_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==3),4);
    familiarity.(foldernames(a).name).suppress_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==4),4);
    familiarity.(foldernames(a).name).clear_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==5),4);
    familiarity.(foldernames(a).name).allold_rt=familiarity.(foldernames(a).name).rt(:,4);
    
    familiarity.(foldernames(a).name).c_maintain_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==1 & familiarity.(foldernames(a).name).rt(:,2)==1),4);
    familiarity.(foldernames(a).name).c_rep_sub_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==2 & familiarity.(foldernames(a).name).rt(:,2)==1),4);
    familiarity.(foldernames(a).name).c_rep_cat_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==3 & familiarity.(foldernames(a).name).rt(:,2)==1),4);
    familiarity.(foldernames(a).name).c_suppress_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==4 & familiarity.(foldernames(a).name).rt(:,2)==1),4);
    familiarity.(foldernames(a).name).c_clear_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,1)==5 & familiarity.(foldernames(a).name).rt(:,2)==1),4);
    familiarity.(foldernames(a).name).c_allold_rt=familiarity.(foldernames(a).name).rt(find(familiarity.(foldernames(a).name).rt(:,2)==1),4);
    
    familiarity.(foldernames(a).name).mean_maintain_rt=mean(familiarity.(foldernames(a).name).maintain_rt);
    familiarity.(foldernames(a).name).mean_rep_sub_rt=mean(familiarity.(foldernames(a).name).rep_sub_rt);
    familiarity.(foldernames(a).name).mean_rep_cat_rt=mean(familiarity.(foldernames(a).name).rep_cat_rt);
    familiarity.(foldernames(a).name).mean_suppress_rt=mean(familiarity.(foldernames(a).name).suppress_rt);
    familiarity.(foldernames(a).name).mean_clear_rt=mean(familiarity.(foldernames(a).name).clear_rt);
    familiarity.(foldernames(a).name).mean_allold_rt=mean(familiarity.(foldernames(a).name).allold_rt);
    
    familiarity.(foldernames(a).name).c_mean_maintain_rt=mean(familiarity.(foldernames(a).name).c_maintain_rt);
    familiarity.(foldernames(a).name).c_mean_rep_sub_rt=mean(familiarity.(foldernames(a).name).c_rep_sub_rt);
    familiarity.(foldernames(a).name).c_mean_rep_cat_rt=mean(familiarity.(foldernames(a).name).c_rep_cat_rt);
    familiarity.(foldernames(a).name).c_mean_suppress_rt=mean(familiarity.(foldernames(a).name).c_suppress_rt);
    familiarity.(foldernames(a).name).c_mean_clear_rt=mean(familiarity.(foldernames(a).name).c_clear_rt);
    familiarity.(foldernames(a).name).c_mean_allold_rt=mean(familiarity.(foldernames(a).name).c_allold_rt);
    
    
    groupfamiliarity.maintain(a)=familiarity.(foldernames(a).name).mean_maintain_rt;
    groupfamiliarity.rep_sub(a)=familiarity.(foldernames(a).name).mean_rep_sub_rt;
    groupfamiliarity.rep_cat(a)=familiarity.(foldernames(a).name).mean_rep_cat_rt;
    groupfamiliarity.suppress(a)=familiarity.(foldernames(a).name).mean_suppress_rt;
    groupfamiliarity.clear(a)=familiarity.(foldernames(a).name).mean_clear_rt;
    groupfamiliarity.allold(a)=familiarity.(foldernames(a).name).mean_allold_rt;
    
    groupfamiliarity.c_maintain(a)=familiarity.(foldernames(a).name).c_mean_maintain_rt;
    groupfamiliarity.c_rep_sub(a)=familiarity.(foldernames(a).name).c_mean_rep_sub_rt;
    groupfamiliarity.c_rep_cat(a)=familiarity.(foldernames(a).name).c_mean_rep_cat_rt;
    groupfamiliarity.c_suppress(a)=familiarity.(foldernames(a).name).c_mean_suppress_rt;
    groupfamiliarity.c_clear(a)=familiarity.(foldernames(a).name).c_mean_clear_rt;
    groupfamiliarity.c_allold(a)=familiarity.(foldernames(a).name).c_mean_allold_rt;
end

clearvars c_data c_stderror data stderror

groupfamiliarity.operation_table=[groupfamiliarity.maintain',groupfamiliarity.rep_sub',groupfamiliarity.rep_cat',groupfamiliarity.suppress',groupfamiliarity.clear'];
groupfamiliarity.c_operation_table=[groupfamiliarity.c_maintain',groupfamiliarity.c_rep_sub',groupfamiliarity.c_rep_cat',groupfamiliarity.c_suppress',groupfamiliarity.c_clear'];

face_color=[[0.741176470588235,0,0.0235294117647059];[0.992156862745098,0.317647058823529,0];[1,0.607843137254902,0.141176470588235];[0,0.584313725490196,0.537254901960784];[0,0.525490196078431,0.768627450980392]];

data=mean(groupfamiliarity.operation_table);
stderror=std(groupfamiliarity.operation_table)/sqrt(length(groupfamiliarity.operation_table));
figure;
hold on;
for x=1:5
i=bar(x,data(x));
set(i,'Facecolor',face_color(x,:))
end
x=1:5;
hold on;errorbar(x,data,stderror,'Color',[0 0 0 ],'LineStyle','none');
title('RT by Operation');
anova1(groupfamiliarity.operation_table);

c_data=mean(groupfamiliarity.c_operation_table);
c_stderror=std(groupfamiliarity.c_operation_table)/sqrt(length(groupfamiliarity.c_operation_table));
figure;
hold on;
for x=1:5
h=bar(x,c_data(x));
set(h,'Facecolor',face_color(x,:));
end
x=1:5;
hold on;errorbar(x,c_data,c_stderror,'Color',[0 0 0 ],'LineStyle','none');
title('RT by Operation - Correct Only');
anova1(groupfamiliarity.c_operation_table);
