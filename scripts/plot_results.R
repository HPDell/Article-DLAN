#!/usr/bin/env Rscript
# R figures for the completed, parameter-matched four-case experiment.
suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(grid)})
args <- commandArgs(trailingOnly=FALSE)
script <- sub('^--file=', '', args[grepl('^--file=',args)])
root <- normalizePath(file.path(dirname(script),'..'))
read <- function(x) read.csv(file.path(root,'evidence',x),check.names=FALSE)
cols <- c('DLAN'='#D55E00','GeoAggregator'='#009E73','GCNNWR'='#0072B2','OLS buffer'='#777777')
set.seed(20261108)
theme_set(theme_bw(base_size=10,base_family='serif') + theme(panel.grid.minor=element_blank(),
 plot.title=element_text(face='bold',size=11),legend.position='bottom',strip.background=element_rect(fill='#F2F2F2')))
save <- function(p,name,w=7,h=4.8) {
 ggsave(file.path(root,'figures',paste0(name,'.pdf')),p,width=w,height=h,device=cairo_pdf)
 ggsave(file.path(root,'figures',paste0(name,'.png')),p,width=w,height=h,dpi=240)
}
m<-read('matched_metrics.csv');te<-subset(m,split=='test');su<-subset(read('matched_summary.csv'),split=='test')
case_order<-c('CO2','OD-Wuhan','WuhanHP','GVI')
for(n in c('m','te','su')) {d<-get(n);d$case<-factor(d$case,levels=case_order);assign(n,d)}
p<-ggplot(te,aes(factor(seed),rmse,colour=model_label,group=model_label))+geom_line(linewidth=.55)+geom_point(size=1.6)+
 facet_wrap(~case,scales='free_y',ncol=2)+scale_colour_manual(values=cols)+
 labs(x='Training seed',y='Test RMSE (case-specific response scale)',colour=NULL,title='All 100 completed fits: 6,957 trainable parameters per model')
save(p,'matched_seed_accuracy',7,5.3)
# Ratios remove response-unit differences without pooling observations across cases.
baseline<-su[su$model_label=='GeoAggregator',c('case','rmse_mean')];names(baseline)[2]<-'geo_rmse'
su<-merge(su,baseline,by='case');su$relative_rmse<-su$rmse_mean/su$geo_rmse
p<-ggplot(su,aes(model_label,relative_rmse,colour=model_label))+geom_hline(yintercept=1,linetype=2,colour='#888888')+
 geom_point(size=3)+facet_wrap(~case,nrow=1)+scale_colour_manual(values=cols)+coord_flip()+
 labs(x=NULL,y='Mean test RMSE / GeoAggregator mean test RMSE',title='Predictive accuracy relative to the same-case baseline')+theme(legend.position='none')
save(p,'matched_relative_accuracy',7,3.3)
p<-ggplot(te,aes(seconds,rmse,colour=model_label))+geom_point(size=2)+facet_wrap(~case,scales='free',ncol=2)+
 scale_colour_manual(values=cols)+labs(x='Fit time (seconds)',y='Test RMSE',colour=NULL,title='Measured accuracy and fit cost; identical training controls')
save(p,'matched_accuracy_cost',7,5.2)
b<-read('matched_bandwidths.csv');b$case<-factor(b$case,levels=case_order)
p<-ggplot(b,aes(axis,bandwidth,group=seed))+geom_line(colour='#999999',alpha=.55)+geom_point(colour='#D55E00',size=1.5)+
 stat_summary(aes(group=1),fun=mean,geom='point',shape=18,size=3,colour='black')+
 facet_wrap(~case,scales='free_x',ncol=2)+labs(x=NULL,y='Bandwidth (model coordinate units)',
 title='Coordinate-specific scales from all 40 DLAN checkpoints',subtitle='Lines: seeds; black diamonds: means')+
 theme(axis.text.x=element_text(angle=20,hjust=1))
save(p,'matched_bandwidths',7,5)
pred<-read('matched_predictions.csv');pred$case<-factor(pred$case,levels=case_order)
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
 scale_fill_manual(values=cols)+labs(x=NULL,y='Squared error from top 1% of flows (%)',title='(a) WuhanOD: concentration of squared error')+theme(legend.position='none')
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
 facet_wrap(~model,ncol=2)+labs(x='Observed GVI',y='Predicted GVI',title='(a) GVI test predictions; seed 1')
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
