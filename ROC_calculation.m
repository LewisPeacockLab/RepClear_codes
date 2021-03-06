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
    maintain_index=find(datastruc.(foldernames(a).name).xdata.matrix(:,5)==1);
    replace_sub_index=find(datastruc.(foldernames(a).name).xdata.matrix(:,5)==2);
    replace_cat_index=find(datastruc.(foldernames(a).name).xdata.matrix(:,5)==3);
    supress_index=find(datastruc.(foldernames(a).name).xdata.matrix(:,5)==4);
    clear_index=find(datastruc.(foldernames(a).name).xdata.matrix(:,5)==5);
    new_img_index=find(datastruc.(foldernames(a).name).xdata.matrix(:,5)==0);
    
    if a==1 || a==2
        resp.(foldernames(a).name).maintain=datastruc.(foldernames(a).name).xdata.matrix(maintain_index,13:14);
        resp.(foldernames(a).name).rep_sub=datastruc.(foldernames(a).name).xdata.matrix(replace_sub_index,13:14);
        resp.(foldernames(a).name).rep_cat=datastruc.(foldernames(a).name).xdata.matrix(replace_cat_index,13:14);
        resp.(foldernames(a).name).suppress=datastruc.(foldernames(a).name).xdata.matrix(supress_index,13:14);
        resp.(foldernames(a).name).clear=datastruc.(foldernames(a).name).xdata.matrix(clear_index,13:14);
        resp.(foldernames(a).name).new_img=datastruc.(foldernames(a).name).xdata.matrix(new_img_index,13:14);
        resp.(foldernames(a).name).all=datastruc.(foldernames(a).name).xdata.matrix(:,13:14);
    else
        resp.(foldernames(a).name).maintain=datastruc.(foldernames(a).name).xdata.matrix(maintain_index,12:13);
        resp.(foldernames(a).name).rep_sub=datastruc.(foldernames(a).name).xdata.matrix(replace_sub_index,12:13);
        resp.(foldernames(a).name).rep_cat=datastruc.(foldernames(a).name).xdata.matrix(replace_cat_index,12:13);
        resp.(foldernames(a).name).suppress=datastruc.(foldernames(a).name).xdata.matrix(supress_index,12:13);
        resp.(foldernames(a).name).clear=datastruc.(foldernames(a).name).xdata.matrix(clear_index,12:13);
        resp.(foldernames(a).name).new_img=datastruc.(foldernames(a).name).xdata.matrix(new_img_index,12:13);
        resp.(foldernames(a).name).all=datastruc.(foldernames(a).name).xdata.matrix(:,12:13);
    end
    cat_type={'maintain','rep_sub','rep_cat','suppress','clear','new_img','all'};
    
    for z=1:length(cat_type)
        new_index=find((resp.(foldernames(a).name).(cat_type{z})(:,1)==1 | resp.(foldernames(a).name).(cat_type{z})(:,1)==2) & resp.(foldernames(a).name).(cat_type{z})(:,2)==1);
        new_index=[new_index;find((resp.(foldernames(a).name).(cat_type{z})(:,1)==3 | resp.(foldernames(a).name).(cat_type{z})(:,1)==4) & resp.(foldernames(a).name).(cat_type{z})(:,2)==0)];
        new_index=sort(new_index);
        new=resp.(foldernames(a).name).(cat_type{z})(new_index,:);
        
        old_index=find((resp.(foldernames(a).name).(cat_type{z})(:,1)==3 | resp.(foldernames(a).name).(cat_type{z})(:,1)==4) & resp.(foldernames(a).name).(cat_type{z})(:,2)==1);
        old_index=[old_index;find((resp.(foldernames(a).name).(cat_type{z})(:,1)==1 | resp.(foldernames(a).name).(cat_type{z})(:,1)==2) & resp.(foldernames(a).name).(cat_type{z})(:,2)==0)];
        old_index=sort(old_index);
        old=resp.(foldernames(a).name).(cat_type{z})(old_index,:);
        
        for i=[4,3,2,1]
            throld=old((old(:,1)>= i),2);
            thrnew=new((new(:,1)>= i),2);
            curve_table(5-i,2)= length(throld) / length(old);
            curve_table(5-i,1)= length(thrnew) / length(new);
        end
        
        curve_table=[0,0;curve_table(:,:)];
        
        ROC.(foldernames(a).name).(cat_type{z}).curve_table=curve_table;
        ROC.(foldernames(a).name).(cat_type{z}).AUC=trapz(ROC.(foldernames(a).name).(cat_type{z}).curve_table(:,1),ROC.(foldernames(a).name).(cat_type{z}).curve_table(:,2));
        clearvars new_index new old_index old curve_table
        
    end
    %figure;
    for z=1:5
        %plot(ROC.(foldernames(a).name).(cat_type{z}).curve_table(:,1),ROC.(foldernames(a).name).(cat_type{z}).curve_table(:,2));
        hold on;
        group.AUC(a,z)=ROC.(foldernames(a).name).(cat_type{z}).AUC;
    end
end

for a=1:length(fieldnames(ROC))
    for z=1:5
        totalROC.(cat_type{z})(a)=ROC.(foldernames(a).name).(cat_type{z}).AUC;
    end
end

figure;violin([totalROC.maintain',totalROC.rep_sub',totalROC.rep_cat',totalROC.suppress',totalROC.clear'])
