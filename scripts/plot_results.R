#!/usr/bin/env Rscript
# Reproducible figures from audited saved predictions. No fitted values invented.
suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(grid)})
args <- commandArgs(trailingOnly=FALSE)
script <- sub('^--file=', '', args[grepl('^--file=',args)])
root <- normalizePath(file.path(dirname(script),'..'))
read <- function(x) read.csv(file.path(root,'evidence',x),check.names=FALSE)
cols <- c('DLAN'='#D55E00','Joint attention'='#0072B2','GeoAggregator'='#009E73','OLS'='#777777',
          'DLAN-P'='#D55E00','GNNWR'='#0072B2','GCNNWR'='#CC79A7')
theme_set(theme_bw(base_size=10,base_family='serif') + theme(panel.grid.minor=element_blank(),
  plot.title=element_text(face='bold',size=11),legend.position='bottom',strip.background=element_rect(fill='#F2F2F2')))
save <- function(p,name,w=7,h=4.8) {
 ggsave(file.path(root,'figures',paste0(name,'.pdf')),p,width=w,height=h,device=cairo_pdf)
 ggsave(file.path(root,'figures',paste0(name,'.png')),p,width=w,height=h,dpi=240)
}
co <- read('co2_metrics.csv');te<-subset(co,split=='test' & model!='ols')
te$model_label<-factor(te$model_label,levels=c('Joint attention','GeoAggregator','DLAN'))
p<-ggplot(te,aes(factor(seed),rmse,colour=model_label,group=model_label))+geom_line(linewidth=.55)+geom_point(size=2)+
 scale_colour_manual(values=cols)+labs(x='Training seed',y='Test RMSE (stored response scale)',colour=NULL,title='CO2: future-year prediction, all ten seeds')
save(p,'co2_seed_accuracy',7,3.8)
sumco<-aggregate(cbind(rmse,mae,r2,seconds,epochs)~model_label,te,mean)
p1<-ggplot(te,aes(seconds,rmse,colour=model_label))+geom_point(size=2,alpha=.75)+scale_colour_manual(values=cols)+
 labs(x='Fit time (seconds)',y='Test RMSE',colour=NULL,title='(a) Accuracy and fitted-run cost')
p2<-ggplot(te,aes(model_label,seconds,fill=model_label))+geom_boxplot(width=.5,outlier.shape=NA)+geom_jitter(width=.09,size=1.2)+
 scale_fill_manual(values=cols)+labs(x=NULL,y='Fit time (seconds)',title='(b) Variation across initialisations')+theme(legend.position='none',axis.text.x=element_text(angle=18,hjust=1))
save((p1+p2)+plot_layout(guides='collect'),'co2_accuracy_cost',7,3.8)
b<-read('co2_bandwidths.csv');b$axis<-factor(b$axis,levels=c('Space x','Space y','Time'))
p<-ggplot(b,aes(axis,bandwidth,group=seed))+geom_line(colour='#999999',alpha=.65)+geom_point(aes(colour=axis),size=2)+
 stat_summary(aes(group=1),fun=mean,geom='point',shape=18,size=4,colour='black')+scale_colour_manual(values=c('#0072B2','#009E73','#D55E00'))+
 labs(x=NULL,y='Bandwidth (model coordinate units)',colour=NULL,title='CO2: learnt coordinate-specific scales',subtitle='Lines: seeds; black diamonds: means')
save(p,'co2_bandwidths',6.7,3.8)
d<-read('co2_predictions_seed1.csv');
p1<-ggplot(d,aes(actual,predicted))+geom_point(alpha=.25,size=.6,colour='#0072B2')+geom_abline(slope=1,intercept=0,linetype=2)+facet_wrap(~model,nrow=1)+
 labs(x='Observed response',y='Predicted response',title='(a) Saved predictions, seed 1')
p2<-ggplot(d,aes(actual,residual))+geom_point(alpha=.25,size=.6,colour='#009E73')+geom_hline(yintercept=0,linetype=2)+facet_wrap(~model,nrow=1)+
 labs(x='Observed response',y='Prediction minus observation',title='(b) Residual structure')
save(p1/p2,'co2_prediction_diagnostics',7,5.5)
h<-read('hp_summary.csv');
p<-ggplot(h,aes(reorder(model,rmse_mean),rmse_mean,colour=model))+geom_errorbar(aes(ymin=rmse_mean-rmse_sd,ymax=rmse_mean+rmse_sd),width=.12)+geom_point(size=3)+
 scale_colour_manual(values=cols)+coord_flip()+labs(x=NULL,y='Test RMSE (stored Price scale)',title='WuhanHP: prediction at unseen housing locations',subtitle='Three training seeds; bars show sample standard deviations')+theme(legend.position='none')
save(p,'hp_accuracy',6.7,3.3)
g<-read('hp_group_errors_seed10.csv');
p1<-ggplot(g,aes(x/1000,y/1000,colour=mean_residual,size=n))+geom_point(alpha=.85)+facet_wrap(~model,nrow=1)+
 scale_colour_gradient2(low='#0072B2',mid='white',high='#D55E00',midpoint=0)+coord_equal()+
 labs(x='Projected easting (km)',y='Projected northing (km)',colour='Mean error',size='Houses',title='(a) Errors at held-out housing locations; seed 10')+theme(legend.position='bottom')
p2<-ggplot(g,aes(n,rmse,colour=model))+geom_point(size=1.5,alpha=.65)+scale_colour_manual(values=cols)+
 labs(x='Number of test houses at location',y='Within-location RMSE',colour=NULL,title='(b) Unequal group size and prediction error')
save(p1/p2,'hp_group_diagnostics',7,6)
o<-read('od_partial_metrics.csv');a<-read('gaussian_ablation.csv')
p1<-ggplot(o,aes(factor(seed),rmse))+geom_col(width=.55,fill='#D55E00')+labs(x='Completed training seed',y='Count RMSE',title='(a) WuhanOD: final DLAN retrieval',subtitle='Four completed seeds; ten-seed comparison pending')
p2<-ggplot(a,aes(dataset,delta_rmse_pct))+geom_col(width=.5,fill='#0072B2')+geom_hline(yintercept=0)+
 labs(x=NULL,y='RMSE increase without Gaussian (%)',title='(b) Gaussian-prior ablation',subtitle='Joint-attention configuration; three paired seeds')
save(p1+p2,'od_and_prior_ablation',7,3.8)
# Exact conceptual scaling curves, deliberately separated from measured runtime.
sc<-expand.grid(N=c(1000,2000,5000,10000,20000,50000),D=c(2,3,4));sc$retrieval<-sc$N*sc$D
p<-ggplot(sc,aes(N,retrieval,colour=factor(D)))+geom_line()+geom_point()+scale_x_log10()+scale_y_log10()+
 labs(x='Number of training references (log scale)',y='Coordinate differences per query (N x D; log scale)',colour='Dimensions',
 title='Exact candidate search: analytical work, not a timing benchmark',subtitle='Attention is restricted to candidates; current exact search still scans the reference bank')
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
