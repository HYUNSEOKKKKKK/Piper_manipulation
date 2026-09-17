// Small CPU geometry kernel. No ROS, GPU, threading or robot interfaces.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

struct Obs {
    int n,c,m,w,h,full_w,full_h,x0,y0;
    const double *rays,*z,*points,*contour;
    const float *depth,*dt;
    const unsigned char *mask;
    double fx,fy,cx,cy,jump,depth_sigma,contour_sigma;
};
struct P {double x,y;};
static double cross(P a,P b,P c){return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);}
static double norm(P a){return std::hypot(a.x,a.y);}
static double clamp(double x,double lo,double hi){return std::max(lo,std::min(hi,x));}
static std::vector<P> hull(std::vector<P> a){
    std::sort(a.begin(),a.end(),[](P x,P y){return x.x==y.x?x.y<y.y:x.x<y.x;});
    std::vector<P> h(16);int k=0;
    for(auto p:a){while(k>=2&&cross(h[k-2],h[k-1],p)<=0)--k;h[k++]=p;}
    for(int i=6,t=k+1;i>=0;--i){auto p=a[i];while(k>=t&&cross(h[k-2],h[k-1],p)<=0)--k;h[k++]=p;}
    h.resize(k-1);return h;
}
static double ray(const double *d,const double *R,const double *t,const double *half,bool &hit){
    double near=-1e100,far=1e100;
    for(int j=0;j<3;++j){
        double v=0,o=0;for(int i=0;i<3;++i){v+=d[i]*R[i*3+j];o-=t[i]*R[i*3+j];}
        if(std::abs(v)<=1e-10)v=v>=0?1e-10:-1e-10;
        double a=(-half[j]-o)/v,b=(half[j]-o)/v;
        near=std::max(near,std::min(a,b));far=std::min(far,std::max(a,b));
    }
    hit=far>=near-1e-6&&near>0;return near;
}
static double sdf(const double *p,const double *R,const double *t,const double *half){
    double sum=0,maxq=-1e100;
    for(int j=0;j<3;++j){double x=0;for(int i=0;i<3;++i)x+=(p[i]-t[i])*R[i*3+j];
        double q=std::abs(x)-half[j];maxq=std::max(maxq,q);sum+=std::max(q,0.)*std::max(q,0.);}
    return std::sqrt(sum)+std::min(maxq,0.);
}
static double distance(P p,const std::vector<P>&h){
    double best=1e100;
    for(size_t i=0;i<h.size();++i){P a=h[i],b=h[(i+1)%h.size()];double x=b.x-a.x,y=b.y-a.y;
        double u=clamp(((p.x-a.x)*x+(p.y-a.y)*y)/std::max(x*x+y*y,1e-8),0,1);
        best=std::min(best,std::hypot(p.x-a.x-u*x,p.y-a.y-u*y));}
    return best;
}
static double quantile(std::vector<double>& a,double q){
    std::sort(a.begin(),a.end());double p=q*(a.size()-1);int lo=static_cast<int>(p),hi=std::min(lo+1,static_cast<int>(a.size()-1));
    return a[lo]+(p-lo)*(a[hi]-a[lo]);
}
extern "C" int cuboid_residual_v1(const Obs *o,const double *R,const double *t,const double *dims,double *out,double *quality){
    double half[3]={dims[0]/2,dims[1]/2,dims[2]/2};std::vector<P> projected;projected.reserve(8);
    for(int sx:{-1,1})for(int sy:{-1,1})for(int sz:{-1,1}){
        double p[3];for(int i=0;i<3;++i)p[i]=t[i]+R[i*3]*half[0]*sx+R[i*3+1]*half[1]*sy+R[i*3+2]*half[2]*sz;
        if(!std::isfinite(p[0])||!std::isfinite(p[1])||!std::isfinite(p[2])||p[2]<=.03)return 0;
        P pixel{o->fx*p[0]/p[2]+o->cx,o->fy*p[1]/p[2]+o->cy};
        if(!std::isfinite(pixel.x)||!std::isfinite(pixel.y)||std::abs(pixel.x)>1e7||std::abs(pixel.y)>1e7)return 0;
        projected.push_back(pixel);
    }
    auto h=hull(projected);if(h.size()<3)return 0;
    int hits=0,visible_count=0;double observed_sum=0,rendered_sum=0;std::vector<double> abs_depth;abs_depth.reserve(o->n);
    for(int i=0;i<o->n;++i){bool hit;double pred=ray(o->rays+3*i,R,t,half,hit);
        double r=hit?pred-o->z[i]:std::abs(sdf(o->points+3*i,R,t,half))+.002;
        out[i]=r/o->depth_sigma/std::sqrt(o->n);hits+=hit;if(quality)abs_depth.push_back(std::abs(r));}
    for(int i=0;i<o->c;++i){double d=distance({o->contour[2*i],o->contour[2*i+1]},h);observed_sum+=d;
        out[o->n+i]=d/o->contour_sigma/std::sqrt(o->c);}
    std::vector<double> lengths(h.size());double perimeter=0;
    for(size_t i=0;i<h.size();++i){P a=h[i],b=h[(i+1)%h.size()];lengths[i]=norm({b.x-a.x,b.y-a.y});perimeter+=lengths[i];}
    size_t edge=0;double cumulative=0;
    for(int i=0;i<o->m;++i){double s=i*perimeter/o->m;while(edge+1<h.size()&&s>=cumulative+lengths[edge])cumulative+=lengths[edge++];
        P a=h[edge],b=h[(edge+1)%h.size()];double u=(s-cumulative)/std::max(lengths[edge],1e-8);
        P p{a.x+u*(b.x-a.x),a.y+u*(b.y-a.y)};double x=p.x-o->x0,y=p.y-o->y0;
        double xc=clamp(x,0,o->w-1.001),yc=clamp(y,0,o->h-1.001);int xi=int(xc),yi=int(yc);double wx=xc-xi,wy=yc-yi;
        double dt=(1-wy)*((1-wx)*o->dt[yi*o->w+xi]+wx*o->dt[yi*o->w+xi+1])+wy*((1-wx)*o->dt[(yi+1)*o->w+xi]+wx*o->dt[(yi+1)*o->w+xi+1]);
        dt+=std::hypot(x-xc,y-yc);
        int px=int(std::nearbyint(x)),py=int(std::nearbyint(y));bool inside=px>=0&&px<o->w&&py>=0&&py<o->h;
        px=std::max(0,std::min(o->w-1,px));py=std::max(0,std::min(o->h-1,py));double meas=o->depth[py*o->w+px];
        double d[3]={(p.x-o->cx)/o->fx,(p.y-o->cy)/o->fy,1};bool hit;double pred=ray(d,R,t,half,hit);
        bool hidden=inside&&!o->mask[py*o->w+px]&&std::isfinite(meas)&&meas>.05&&meas+o->jump<pred;
        bool clipped=p.x<1||p.x>=o->full_w-2||p.y<1||p.y>=o->full_h-2;
        bool visible=!hidden&&!clipped;double value=visible?dt:0.;visible_count+=visible;rendered_sum+=value;
        out[o->n+o->c+i]=value/o->contour_sigma/std::sqrt(o->m);
    }
    if(quality){quality[0]=quantile(abs_depth,.5);quality[1]=quantile(abs_depth,.9);quality[2]=double(hits)/o->n;
        quality[3]=visible_count?(observed_sum/o->c+rendered_sum/visible_count)/2:1e6;quality[4]=double(visible_count)/o->m;}
    return 1;
}
