?
[xh, xcrit_p, xadj_ci_cvrg, xadj_p] = fdr_bh(pvalue_matrix, args.alpha, 'pdep', 'no');
    
multi_out = sprintf('critical p=%1.4f, CI coverage=%1.4f', xcrit_p, xadj_ci_cvrg);
    
%%
format shortG
?
xmtcmp = 'tukey-kramer';
xnames = {'Target','RelatedNontarget','Nontarget'};
n_targ = length(xnames);
?
for xtarg = 1:n_targ
    
    clear xcorr
    xcorr = [];
    
    for xcond = 1:n_condition
        t_corr = [];
        
        for xtr = args.percept_win
            t_corr = horzcat(t_corr, grp_rsa{xph}.timecourse.cond{xcond}.targ{xtarg}.tr{xtr}.corr_w');
        end
        
        xcorr = horzcat(xcorr, mean(t_corr, 2));
    end
    
    grp_rsa{xph}.percept.mean_w(:, xtarg) = mean(xcorr, 2);
end
?
%*************** ANOVA
% xtarg:  1_target, 2_related_nontarg 2_nontarg
?
clear it_matrix xpvalue
?
xtable_rsa = array2table(grp_rsa{xph}.percept.mean_w, 'VariableNames', xnames);
?
xanova     = {'targets'};
xmeasures  = table((1:n_targ)','VariableNames', xanova);
xrepmeas   = fitrm(xtable_rsa, sprintf('%s-%s~1', xnames{1}, xnames{end}),'WithinDesign',xmeasures);
xanova_out = ranova(xrepmeas);
?
xdiff_anova = sprintf('one-way ANOVA: F(%s, %s)=%2.4f, p=%1.4f', ...
    num2str(xanova_out.DF(1)), num2str(xanova_out.DF(2)),...
    xanova_out.F(1), xanova_out.pValue(1));
?
xpvalue_table = multcompare(xrepmeas, xanova,'ComparisonType', xmtcmp);%'bonferroni'
?
%*************** ttest
xpvalue = nan(n_targ-1, n_targ-1);
?
for xrow = 1:(n_targ-1)
    for xcol = xrow+1:n_targ
        xunit = (xcol-1) + ((n_targ-1) * (xrow-1));
        xpvalue(xrow, xcol) = xpvalue_table.pValue(xunit);
    end
end
?
grp_rsa{xph}.percept.anova    = xanova_out;
grp_rsa{xph}.percept.multcomp = xpvalue_table;
?
%% ============= ONE-SAMPLE TTEST
cond_names = {'Maintain','RepCategory','RepSubcate','Suppress','Clear'};
?
xacc_csv    = fullfile(dirs.mvpa.group.out{xph}, sprintf('grp_acc_matrix_%s_n%s.csv', basename, num2str(length(xsub_groups))));
xtable      = readtable(xacc_csv);
xacc_matrix = table2array(xtable);
?
for xsub = 1:length(xsub_groups)
    xcol = n_condition + 1;
    xacc_matrix(xsub, xcol) = mean(xacc_matrix(xsub, 1:n_condition));
end
?
%*************** one-sample ttest
xchance = 1/n_condition;
for xcond = 1:(n_condition+1)
    [~, p(xcond)] = ttest(xacc_matrix(:,xcond), xchance);
end
?
xmean_table = array2table([mean(xacc_matrix); std(xacc_matrix)/sqrt(length(xsub_groups)); p],...
    'VariableNames', [cond_names, 'total']);%, 'RowNames', {'mean','se','ttest'});
?
%*************** write tables to csv files
csv_name = sprintf('%s/classifier_acc_ttest_%s_n%s.csv', xoutput_dir, basename, num2str(length(xsub_groups)));
writetable(xmean_table, csv_name, 'WriteRowNames', true)
?
%% ============= ONE-WAY ANOVA
?
xmtcmp = 'tukey-kramer';
xnames = {'Target','RelatedNontarget','Nontarget'};
n_targ = length(xnames);
?
for xtarg = 1:n_targ
    
    clear xcorr
    xcorr = [];
    
    for xcond = 1:n_condition
        t_corr = [];
        
        for xtr = args.percept_win
            t_corr = horzcat(t_corr, grp_rsa{xph}.timecourse.cond{xcond}.targ{xtarg}.tr{xtr}.corr_w');
        end
        
        xcorr = horzcat(xcorr, mean(t_corr, 2));
    end
    
    grp_rsa{xph}.percept.mean_w(:, xtarg) = mean(xcorr, 2);
end
?
%*************** ANOVA
% xtarg:  1_target, 2_related_nontarg 2_nontarg
?
clear it_matrix xpvalue
?
xtable_rsa = array2table(grp_rsa{xph}.percept.mean_w, 'VariableNames', xnames);
?
xanova     = {'targets'};
xmeasures  = table((1:n_targ)','VariableNames', xanova);
xrepmeas   = fitrm(xtable_rsa, sprintf('%s-%s~1', xnames{1}, xnames{end}),'WithinDesign',xmeasures);
xanova_out = ranova(xrepmeas);
?
xdiff_anova = sprintf('one-way ANOVA: F(%s, %s)=%2.4f, p=%1.4f', ...
    num2str(xanova_out.DF(1)), num2str(xanova_out.DF(2)),...
    xanova_out.F(1), xanova_out.pValue(1));
?
xpvalue_table = multcompare(xrepmeas, xanova,'ComparisonType', xmtcmp);%'bonferroni'
?
%*************** ttest
xpvalue = nan(n_targ-1, n_targ-1);
?
for xrow = 1:(n_targ-1)
    for xcol = xrow+1:n_targ
        xunit = (xcol-1) + ((n_targ-1) * (xrow-1));
        xpvalue(xrow, xcol) = xpvalue_table.pValue(xunit);
    end
end
?
grp_rsa{xph}.percept.anova    = xanova_out;
grp_rsa{xph}.percept.multcomp = xpvalue_table;
?
%% ============= TWO-WAY ANOVA: within evidence (perception/prediciton)
%*************** condition * pair (2 x 3|4) for target with xfull_array
clear xfactor xarray
?
fprintf('\n\n################################################\n');
fprintf('#### 2-way ANOVA %s: condition * pair\n', evidence_name{xregs});
fprintf('################################################\n');
?
xfactor = {'condition','pair'};
xarray  = xfull_array(getDATA(xfull_array, xfactorial_vars, {'target'}, {1}), :);
?
Y  = xarray(:, findCol(xfactorial_vars, {'measure'}));
S  = xarray(:, findCol(xfactorial_vars, {'subject'}));
F1 = xarray(:, findCol(xfactorial_vars, {'condition'}));
F2 = xarray(:, findCol(xfactorial_vars, {'pair'}));
?
xstats = rm_anova2(Y, S, F1, F2, xfactor);
xtable = cell2table(xstats);
?
xgrp_distribution{xregs}.anova.CondxPair.factors = xfactor;
xgrp_distribution{xregs}.anova.CondxPair.table   = xtable;
?
for xmain = 1:length(xfactor)
    xunit     = xmain + 1;
    xtx_stats = sprintf('%s: F(%s,%s)=%4.4f, p=%4.4f', ...
        xfactor{xmain}, num2str(xtable.xstats3{xunit}), ...
        num2str(xtable.xstats3{length(xtable.xstats1)}), ...
        xtable.xstats5{xunit}, xtable.xstats6{xunit});
    
    fprintf('Main effect: %s\n', xtx_stats)
    
    xgrp_distribution{xregs}.anova.CondxPair.stats.main{xmain} = xtx_stats;
    xgrp_distribution{xregs}.anova.CondxPair.pvalue.main{xmain} = xtable.xstats6{xunit};
end
?
xunit     = 4;
xtx_stats = sprintf('%s x %s: F(%s,%s)=%4.4f, p=%4.4f', ...
    xfactor{1}, xfactor{2}, num2str(xtable.xstats3{xunit}), ...
    num2str(xtable.xstats3{length(xtable.xstats1)}), ...
    xtable.xstats5{xunit}, xtable.xstats6{xunit});
?
fprintf('Interaction: %s\n', xtx_stats)
?
xgrp_distribution{xregs}.anova.CondxPair.stats.interact  = xtx_stats;
xgrp_distribution{xregs}.anova.CondxPair.pvalue.interact = xtable.xstats6{xunit};
?
%% ============= THREE-WAY ANOVA: within evidence (perception/prediciton)
%*************** condition * pair * target (2 x 3|4 x 2)
%*************** RMAOV33(X, alpha, varnames, xfile)
%     X - data matrix (Size of matrix must be n-by-5
%         column 1: dependent variable
%         column 2: independent variable 1 (within subjects)
%         column 3: independent variable 2 (within subjects)
%         column 4: independent variable 3 (within subjects)
%         column 5: subject.
% alpha - significance level (default = 0.05).
?
fprintf('\n\n################################################\n');
fprintf('#### 3 WAY ANOVA: condition * pair * target: %s\n', evidence_name{xregs});
fprintf('################################################\n');
?
clear xfactor xarray
?
xfactor = {'condition','pair','target','subject'};
xarray  = zeros(size(xfull_array, 1), length(xfactor)+1);
?
xarray(:,1) = xfull_array(:, findCol(xfactorial_vars, {'measure'}));
?
for xfact = 1:length(xfactor)
    xarray(:,xfact + 1) = xfull_array(:, findCol(xfactorial_vars, {xfactor(xfact)}));
end
?
RMAOV33(xarray, 0.05, xfactor);
%% ============= FACTORIAL ANOVA: 3-way repeated measure ANOVA
%*************** RMAOV33(X, alpha, varnames, xfile)
%     X - data matrix (Size of matrix must be n-by-5;dependent variable=column 1;
%         independent variable 1 (within subjects)=column 2;independent variable 2
%         (within subjects)=column 3; independent variable 3 (within subjects)
%         =column 4; subject=column 5).
% alpha - significance level (default = 0.05).
?
fprintf('############################################\n');
fprintf('#### 3 WAY ANOVA: cueitem * condition * pair\n');
fprintf('############################################\n');
?
for xmes = 1:2
    fprintf('\n*************** %s\n', meas_name{xmes});
    
    X        = rm_matrix{xmes};%[rm_matrix{xmes} item_fact cond_fact pair_fact xsubjects];
    varnames = {'cueitem','condition','pair','subject'};
    
    RMAOV33(X, 0.05, varnames);
    
    diary on
end
?
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%*************** DESCRIPTIVE STATS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ============= ENCODING ACCURACY & RT
%*************** 1. overall mean & se
diary off;
diary(xoutput_txt);
?
clear xmean xsem
?
fprintf('\n\n');
fprintf('############################################\n');
fprintf('#### OVERALL DESCRIPTIVE STATS\n');
fprintf('############################################\n');
?
for xsub = 1:args.n_sub
    itdata = xmatrix{xsub};
    
    clear sub_data
    sub_data = itdata(:,findCol(xheader,{'correct','rt'}));
    
    xacc_all(xsub) = mean(sub_data(:,1));
    xrt_all(xsub)  = mean(sub_data(sub_data(:,1)==1, 2));%RT: only for the correct trials
end
?
xmean{acc} = mean(xacc_all);
xsem{acc}  = std(xacc_all)/sqrt(args.n_sub);
?
xmean{rt} = mean(xrt_all);
xsem{rt}  = std(xrt_all)/sqrt(args.n_sub);
?
%*************** writing acc & rt
for xmes = 1:2 %1_acc, 2_rt
    fprintf('*************** Overall (cue+item+np) %s\n', meas_name{xmes});
    fprintf('%s: Mean: %1.4f, SEM: %1.4f\n\n',meas_name{xmes}, xmean{xmes}, xsem{xmes});
    
    diary on
end
?
%% ============= VIOLIN PLOTTING
% violin: all conditions
% box: cueitem/conditions/
xcolor{1} = [240, 124, 33]/255;% congruent
xcolor{2} = [0, 167, 169]/255;% incongruent
xcolor{3} = [144, 144, 144]/255;% baseline
?
%*************** accuracy
fig_rect = [0 0 800 700];
?
xfig = figure;
set(xfig, 'Position', fig_rect)
?
%*************** violin
y_lim     = [0.2 1.3];
y_tick    = y_lim(1):0.1:y_lim(2);
x_tick    = 1:17;
?
for xitem = 1:2
    for xcond = 1:n_conds
        for xpair = 1:n_pairs
            xunit = xpair + (n_pairs * (xcond-1)) + ...
                ((n_pairs * n_conds) * (xitem-1));
            it_colors(xunit, :) = xcolor{xcond};
            it_labels{xunit} = sprintf('%s %s pair%d', ...
                item_names{xitem}, conditions{xcond}, xpair);
        end
    end
end
?
%*************** np
it_colors(17,:) = xcolor{3};
it_labels{17}   = 'np';
?
%*************** legend
h    = zeros(3, 1);
for xcond=1:3
    h(xcond) = plot(NaN, NaN, 'Color', xcolor{xcond},'LineWidth',5);
    hold on;
end
?
clear xlegend
xlegend        = {'congruent','incongruent','np'};
lg             = legend(xlegend);
lg.Location    = 'BestOutside';
legend(xlegend,'AutoUpdate','off')
?
%*************** violin
violin([xacc{1}(:,1:8) xacc{2}(:,:)],'facecolor',it_colors,'xlabel',it_labels,...
    'edgecolor','','medc','','mc','k','plotlegend',''); hold on
?
title(sprintf('encoding accuracy (N=%s)', num2str(args.n_sub)));
?
set(gca,'ylim', y_lim)
set(gca,'YTick', y_tick)
set(gca,'XTick', x_tick, 'XTickLabel', it_labels)
xtickangle(90)
xlabel('condition');
ylabel('accuracy');
grid on
?
%%============== LINEAR FITTING
%*************** cue congruent
clear xx yy
?
for xitem = 1:2
    for xcond = 1:2
        xx = []; yy = [];
        
        for i = 1:4
            xunit = i + (4 * (xcond-1)) + (8 * (xitem-1));
            xx = horzcat(xx, ones(1, args.n_sub)*xunit);
            
            xunit = i + (4 * (xcond-1));
            yy = horzcat(yy, xacc{xitem}(:,xunit)');
        end
        
        [p, stat] = polyfit(xx,yy,1);
        lm        = fitlm(xx,yy,'linear');
        f         = polyval(p,xx);
        plot(xx,f,'-','Color',xcolor{xcond},'LineWidth',2)
        
        x_posit = 2 + (4 * (xcond-1)) + (8 * (xitem-1));
        text(x_posit, y_lim(2) - 0.2, ...
            sprintf('y=%1.2fx + %1.2f\n p=%1.4f',p(1),p(2),lm.coefTest),...
            'FontSize', 10, 'FontWeight', 'bold');
    end
end
?
%*************** SAVE FIGURE
fig_fname = sprintf('%s/violin_plot_encoding_accuracy_all', xout_dir);
?
savefig(xfig, sprintf('%s.fig', fig_fname));
set(gcf,'PaperUnits','inches','PaperPosition',fig_rect/100)
print('-djpeg', sprintf('%s.jpg',fig_fname), '-r100')
saveas(xfig, sprintf('%s.jpg',fig_fname), 'jpg')
?
close(xfig);
?
%% ============= BOX PLOTTING
% violin: all conditions
% box: cueitem/conditions/
xcolor{1} = [240, 124, 33]/255;% congruent
xcolor{2} = [0, 167, 169]/255;% incongruent
xcolor{3} = [144, 144, 144]/255;% baseline
?
for xitem = 1:2
    for xcond = 1:n_conds
        for xpair = 1:n_pairs
            xunit = xpair + (n_pairs * (xcond-1)) + ...
                ((n_pairs * n_conds) * (xitem-1));
            it_colors(xunit, :) = xcolor{xcond};
            it_labels{xunit} = sprintf('%s %s pair%d', ...
                item_names{xitem}, conditions{xcond}, xpair);
        end
    end
end
?
it_colors(17,:) = xcolor{3};
it_labels{17}   = 'np';