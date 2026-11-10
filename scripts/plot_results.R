#!/usr/bin/env Rscript
# R figures for the completed, parameter-matched four-case experiment.
suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(grid)})
args <- commandArgs(trailingOnly=FALSE)
script <- sub('^--file=', '', args[grepl('^--file=',args)])
root <- normalizePath(file.path(dirname(script),'..'))
read <- function(x) read.csv(file.path(root,'evidence',x),check.names=FALSE)
cols <- c('DLAN'='#D55E00','GeoAggregator'='#009E73','GCNNWR'='#0072B2','OLS buffer'='#777777','GNNWR'='#CC79A7','GTNNWR'='#E69F00')
set.seed(20261108)
theme_set(theme_bw(base_size=10,base_family='serif') + theme(panel.grid.minor=element_blank(),
 plot.title=element_text(face='bold',size=11),legend.position='bottom',strip.background=element_rect(fill='#F2F2F2')))
save <- function(p,name,w=7,h=4.8) {
 ggsave(file.path(root,'figures',paste0(name,'.pdf')),p,width=w,height=h,device=cairo_pdf)
 ggsave(file.path(root,'figures',paste0(name,'.png')),p,width=w,height=h,dpi=240)
}
m<-read('reported_metrics.csv');te<-subset(m,split=='test');su<-subset(read('reported_summary.csv'),split=='test')
case_order<-c('CO2','OD-Wuhan','WuhanHP','GVI')
for(n in c('m','te','su')) {d<-get(n);d$case<-factor(d$case,levels=case_order);assign(n,d)}
p<-ggplot(te,aes(factor(seed),rmse,colour=model_label,group=model_label))+geom_line(linewidth=.55)+geom_point(size=1.6)+
 facet_wrap(~case,scales='free_y',ncol=2)+scale_colour_manual(values=cols)+
 labs(x='Training seed',y='Test RMSE (case-specific response scale)',colour=NULL,title='Reported configurations: WuhanOD width 12 / K512 / 7,069 parameters')
save(p,'matched_seed_accuracy',7,5.3)
# Figure 2: seed-level distributions, with one dataset per row and metric per column.
accuracy_long <- rbind(
 transform(te, metric='RMSE', value=rmse),
 transform(te, metric='MAE', value=mae))
accuracy_long$metric <- factor(accuracy_long$metric, levels=c('RMSE','MAE'))
accuracy_long$model_label <- factor(accuracy_long$model_label,
 levels=c('DLAN','GeoAggregator','GTNNWR','GNNWR','GCNNWR'))
case_labels <- c('CO2'='CO[2]','OD-Wuhan'='WuhanOD','WuhanHP'='WuhanHP','GVI'='GVI')
p <- ggplot(accuracy_long,aes(model_label,value,colour=model_label,fill=model_label)) +
 geom_boxplot(width=.55,alpha=.18,outlier.shape=NA,linewidth=.45) +
 geom_point(position=position_jitter(width=.10,height=0,seed=20261110),
            size=1.25,alpha=.8) +
 facet_grid(case ~ metric,scales='free_y',
            labeller=labeller(case=as_labeller(case_labels,default=label_parsed))) +
 scale_colour_manual(values=cols) + scale_fill_manual(values=cols) +
 scale_y_continuous(expand=expansion(mult=c(.04,.09))) +
 labs(x='Model',y='Test metric value') +
 theme(legend.position='none',axis.text.x=element_text(angle=25,hjust=1),
       strip.text=element_text(size=10),panel.spacing=unit(.75,'lines'))
save(p,'matched_relative_accuracy',7,8.5)
p<-ggplot(te,aes(seconds,rmse,colour=model_label))+geom_point(size=2)+facet_wrap(~case,scales='free',ncol=2)+
 scale_colour_manual(values=cols)+labs(x='Fit time (seconds)',y='Test RMSE',colour=NULL,title='Recorded accuracy and fit cost; WuhanOD runs span two machines')
save(p,'matched_accuracy_cost',7,5.2)
b<-read('reported_bandwidths.csv');b$case<-factor(b$case,levels=case_order)
p<-ggplot(b,aes(axis,bandwidth,group=seed))+geom_line(colour='#999999',alpha=.55)+geom_point(colour='#D55E00',size=1.5)+
 stat_summary(aes(group=1),fun=mean,geom='point',shape=18,size=3,colour='black')+
 facet_wrap(~case,scales='free_x',ncol=2)+labs(x=NULL,y='Bandwidth (model coordinate units)',
 title='Coordinate-specific scales from all 40 DLAN checkpoints',subtitle='Lines: seeds; black diamonds: means')+
 theme(axis.text.x=element_text(angle=20,hjust=1))
save(p,'matched_bandwidths',7,5)
pred<-read('reported_predictions.csv');pred$case<-factor(pred$case,levels=case_order)
# Identifier-fixed example, not selected by prediction quality.
co<-subset(pred,case=='CO2' & seed==1)
p1<-ggplot(co,aes(actual,predicted))+geom_point(size=.6,alpha=.3,colour='#0072B2')+geom_abline(slope=1,intercept=0,linetype=2)+facet_wrap(~model,nrow=1)+labs(x='Observed response',y='Predicted response',title='(a) CO2: seed 1')
p2<-ggplot(co,aes(actual,residual))+geom_point(size=.6,alpha=.3,colour='#009E73')+geom_hline(yintercept=0,linetype=2)+facet_wrap(~model,nrow=1)+labs(x='Observed response',y='Prediction minus observation',title='(b) CO2: residual structure')
save(p1/p2,'matched_co2_predictions',7,5.5)
g<-subset(read('matched_hp_groups.csv'),split=='test')
# Each group receives equal weight here; ten-seed mean errors are mapped.
groups<-aggregate(cbind(rmse,mean_residual,n,x,y)~model+group,g,mean)
p1<-ggplot(groups,aes(x/1000,y/1000,colour=mean_residual,size=n))+geom_point(alpha=.8)+facet_wrap(~model,nrow=1)+coord_equal()+
 scale_colour_gradient2(low='#0072B2',mid='white',high='#D55E00',midpoint=0)+labs(x='Projected easting (km)',y='Projected northing (km)',colour='Mean error',size='Houses',title='(a) WuhanHP: ten-seed mean errors at held-out locations')
p2<-ggplot(groups,aes(n,rmse,colour=model))+geom_point(size=1.6,alpha=.7)+scale_colour_manual(values=cols)+
 labs(x='Houses at test location',y='Mean within-location RMSE',colour=NULL,title='(b) Prediction errors and unequal group sizes')
save(p1/p2,'matched_hp_groups',7,5.8)
od<-subset(te,case=='OD-Wuhan')
p1<-ggplot(od,aes(model_label,upper_1pct_sse_share*100,fill=model_label))+geom_boxplot(width=.4,outlier.shape=NA)+geom_jitter(width=.07,size=1.4)+
 scale_fill_manual(values=cols)+labs(x=NULL,y='Squared error from top 1% of flows (%)',title='(a) WuhanOD: reported configuration error concentration')+theme(legend.position='none')
p2<-ggplot(subset(pred,case=='OD-Wuhan' & seed==1),aes(log1p(actual),log1p(predicted)))+
 geom_point(size=.7,alpha=.3,colour='#0072B2')+geom_abline(slope=1,intercept=0,linetype=2)+facet_wrap(~model,nrow=1)+
 labs(x='log(1 + observed count)',y='log(1 + predicted count)',title='(b) Same test interactions; seed 1')
save(p1/p2,'matched_od_tail',7,5.8)
gv<-subset(pred,case=='GVI');neural<-subset(gv,model!='OLS buffer')
# Observed-response bins are descriptive diagnostics, not fit-time partitions.
breaks<-unique(quantile(neural$actual,probs=seq(0,1,.2)))
neural$bin<-cut(neural$actual,breaks=breaks,include.lowest=TRUE,labels=FALSE)
q<-aggregate(cbind(actual,predicted)~model+seed+bin,neural,mean)
qa<-aggregate(cbind(actual,predicted)~model+bin,q,mean)
p1<-ggplot(subset(gv,seed==1),aes(actual,predicted))+geom_point(size=1.1,alpha=.45,colour='#0072B2')+geom_abline(slope=1,intercept=0,linetype=2)+
 facet_wrap(~model,ncol=3)+labs(x='Observed GVI',y='Predicted GVI',title='(a) GVI test predictions; seed 1')
p2<-ggplot(qa,aes(actual,predicted,colour=model,group=model))+geom_line()+geom_point()+geom_abline(slope=1,intercept=0,linetype=2)+
 scale_colour_manual(values=cols)+labs(x='Mean observed GVI in response quintile',y='Mean prediction across ten seeds',colour=NULL,title='(b) Response-dependent prediction; held-out observations')
save(p1/p2,'matched_gvi_response',7,6.5)
a<-subset(read('gaussian_ablation.csv'),split=='test')
p<-ggplot(a,aes(dataset,delta_rmse_pct))+geom_col(width=.45,fill='#0072B2')+geom_hline(yintercept=0)+
 labs(x=NULL,y='Test RMSE increase without Gaussian (%)',title='Historical joint-attention Gaussian-prior ablation',subtitle='Three paired seeds; distinct from the 6,957-parameter main experiment')
save(p,'historical_gaussian_ablation',6.7,3.5)
sc<-expand.grid(N=c(1000,2000,5000,10000,20000,50000),D=c(2,3,4));sc$retrieval<-sc$N*sc$D
p<-ggplot(sc,aes(N,retrieval,colour=factor(D)))+geom_line()+geom_point()+scale_x_log10()+scale_y_log10()+
 labs(x='Training references (log scale)',y='Coordinate differences per query (N x D; log scale)',colour='Dimensions',title='Exact candidate search: analytical work, not measured timing')
save(p,'retrieval_complexity',6.7,3.7)

# Supplementary protocols remain separate from the primary 6,957-parameter scores.
sm<-read('supplement_metrics.csv');st<-subset(sm,split=='test')
cap<-subset(st,experiment=='OD capacity')
base<-subset(read('matched_metrics.csv'),case=='OD-Wuhan' & split=='test');base$budget<-'6,957 parameters'
cap$budget<-'7,069 parameters'
combined<-rbind(base[,c('seed','model_label','rmse','budget')],cap[,c('seed','model_label','rmse','budget')])
p<-ggplot(combined,aes(model_label,rmse,colour=model_label))+geom_boxplot(width=.45,outlier.shape=NA)+geom_jitter(width=.07,size=1.5)+
 stat_summary(fun=mean,geom='point',shape=18,size=3,colour='black')+facet_wrap(~budget,nrow=1)+scale_colour_manual(values=cols)+
 labs(x=NULL,y='Count-scale test RMSE',colour=NULL,title='WuhanOD: capacity allocation and seed variability',subtitle='Black diamonds: means; each budget has its own complete comparison')+
 theme(axis.text.x=element_text(angle=15,hjust=1),legend.position='none')
save(p,'od_capacity_stability',7,4)
w12<-subset(st,experiment=='OD capacity' & model_label=='DLAN');w12$configuration<-'Width 12'
w16<-subset(st,experiment=='OD width16');w16$configuration<-'Width 16'
w<-rbind(w12[,c('seed','rmse','configuration')],w16[,c('seed','rmse','configuration')])
p1<-ggplot(w,aes(factor(seed),rmse,colour=configuration,group=configuration))+geom_line()+geom_point()+
 scale_colour_manual(values=c('#D55E00','#0072B2'))+labs(x='Seed',y='Count RMSE',colour=NULL,title='(a) Attention width; K = 1,024')
k<-subset(st,experiment=='OD neighbours');k$configuration<-k$model_label
w12$configuration<-'K1024'
k<-rbind(k[,c('seed','rmse','configuration')],w12[,c('seed','rmse','configuration')])
k$configuration<-factor(k$configuration,levels=c('K512','K1024','K2048'))
p2<-ggplot(k,aes(configuration,rmse,fill=configuration))+geom_boxplot(width=.45,outlier.shape=NA)+geom_jitter(width=.08,size=1.4)+
 scale_fill_manual(values=c('#0072B2','#D55E00','#009E73'))+labs(x=NULL,y='Count RMSE',title='(b) Candidate count; width = 12',subtitle='K512/K2048 were trained on another machine')+theme(legend.position='none')
save(p1/p2,'od_width_neighbours',7,5.7)
g3<-subset(st,experiment=='GVI 3D');g3$dimension<-'3D'
g2<-subset(te,case=='GVI' & model_label %in% c('DLAN','GeoAggregator'));g2$dimension<-'2D'
gc<-rbind(g3[,c('seed','model_label','rmse','dimension')],g2[,c('seed','model_label','rmse','dimension')])
gc$dimension<-factor(gc$dimension,levels=c('2D','3D'))
p<-ggplot(gc,aes(dimension,rmse,group=seed))+geom_line(colour='#aaaaaa',alpha=.7)+geom_point(aes(colour=dimension),size=2)+
 stat_summary(aes(group=1),fun=mean,geom='point',shape=18,size=4,colour='black')+facet_wrap(~model_label,nrow=1)+
 scale_colour_manual(values=c('#0072B2','#D55E00'))+labs(x=NULL,y='GVI test RMSE',colour='Coordinates',title='GVI: identical spatial split, with or without capture month',subtitle='Widths adjusted to maintain 6,957 parameters; DLAN candidate budget also changes')
save(p,'gvi_2d_3d',7,4)

# Journal figure: vector architecture drawn in R; labels correspond to code.
architecture<-function() {
 grid.newpage()
 box<-function(x,y,w,h,label,fill) {
  grid.roundrect(x,y,w,h,r=unit(.02,'snpc'),gp=gpar(fill=fill,col='#4D4D4D'))
  grid.text(label,x,y,gp=gpar(fontfamily='serif',fontsize=10))
 }
 arr<-function(x1,y1,x2,y2)grid.lines(c(x1,x2),c(y1,y2),arrow=arrow(length=unit(1.8,'mm')),gp=gpar(col='#555555'))
 grid.text('Dimension-specific Local Attention Network',.5,.95,gp=gpar(fontfamily='serif',fontface='bold',fontsize=14))
 box(.14,.76,.23,.20,'Query: x, z\nTraining bank: x, z, y\nExclude query row','#E6EFF6')
 box(.48,.76,.32,.20,'Bandwidth-scaled retrieval\nTop-K per block and head\nID union; refresh per microbatch','#E2F2EA')
 box(.84,.76,.24,.20,'Shared feature and\nFourier encoders\nResponse in values only','#E6EFF6')
 arr(.255,.76,.32,.76);arr(.64,.76,.72,.76)
 box(.50,.43,.82,.22,'Axis-specific Q / K / V projections\nMean content score + summed geographic biases + log Gaussian product\nONE softmax over shared candidates; pool and fuse axis values','#E2F2EA')
 arr(.84,.66,.84,.54)
 box(.50,.14,.59,.18,'Residual attention blocks -> local condition h\nFiLM: GELU((1 + tanh gamma(h)) a + beta(h))\nShared nonlinear predictor of query covariates -> prediction','#F4EAF2')
 arr(.50,.32,.50,.23)
 grid.lines(c(.14,.14,.205),c(.66,.14,.14),arrow=arrow(length=unit(1.8,'mm')),gp=gpar(col='#555555'))
 grid.text('Query x',.105,.27,rot=90,gp=gpar(fontfamily='serif',fontsize=9))
}
cairo_pdf(file.path(root,'figures/architecture.pdf'),width=7,height=4.8);architecture();dev.off()
png(file.path(root,'figures/architecture.png'),width=1680,height=1152,res=240);architecture();dev.off()
writeLines(capture.output(sessionInfo()),file.path(root,'evidence/R_sessionInfo.txt'))
