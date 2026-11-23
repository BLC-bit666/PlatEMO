// Exact arithmetic kernels. Build without fast-math or fused multiply-add.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

extern "C" void route_costs(int n, const double* x, const double* length,
    const double* km, const double* climb, const double* surface, const double* road,
    const uint8_t* mode, double* out)
{
    const double boarding = (0.5*x[4])*1000.0;
    for (int i=0;i<n;++i) {
        if (mode[i]==1) out[i]=length[i]+(x[3]*km[i])*1000.0;
        else if (mode[i]==2) out[i]=length[i]+boarding;
        else {
            double v=length[i]+(climb[i]*x[0])*1000.0;
            v=v+((surface[i]*x[1])*km[i])*1000.0;
            out[i]=v+((road[i]*x[2])*km[i])*1000.0;
        }
    }
}

extern "C" int target_astar(int n, const int32_t* ptr, const int32_t* indices,
    const double* weights, int source, int target, const double* lower,
    double factor, double* dist, int32_t* pred)
{
    const double inf=std::numeric_limits<double>::infinity();
    std::fill(dist,dist+n,inf);std::fill(pred,pred+n,-9999);
    using Item=std::tuple<double,double,int>;
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> queue;
    // Discount bounds for positive-sum rounding in both traversal directions.
    factor*=1.0-16.0*n*std::numeric_limits<double>::epsilon();
    auto h=[&](int v){return factor>0 ? factor*lower[v] : 0.0;};
    queue.emplace(h(source),0.0,source);dist[source]=0.0;
    double cutoff=inf;int visited=0;
    while(!queue.empty()) {
        auto [estimate,du,u]=queue.top();queue.pop();
        if(estimate>cutoff) break;
        if(du!=dist[u]) continue;
        ++visited;
        if(u==target) {
            double scale=std::max(du,1.0);
            cutoff=du+8.0*n*(std::nextafter(scale,inf)-scale);
        }
        for(int32_t e=ptr[u];e<ptr[u+1];++e) {
            int v=indices[e];double dv=du+weights[e];
            if(dv<dist[v]) {dist[v]=dv;pred[v]=u;queue.emplace(dv+h(v),dv,v);}
        }
    }
    return visited;
}

extern "C" void qr_blend(int pixels,const float* alpha,const uint8_t* reference,
    const uint8_t* code,uint8_t* out)
{
    for(int p=0;p<pixels;++p) {
        // NumPy computes 1-alpha in float32, then both products in float64.
        const float inverse=1.0f-alpha[p];
        for(int c=0;c<3;++c) {
            int k=3*p+c;
            double value=double(inverse)*reference[k]+double(alpha[p])*code[k];
            out[k]=static_cast<uint8_t>(std::nearbyint(value));
        }
    }
}

extern "C" double qr_mse(int n,const uint8_t* image,const uint8_t* reference)
{
    // Integer pixels imply exact integer squared errors; the total is <2^53.
    uint64_t sum=0;
    for(int i=0;i<n;++i) {int d=int(image[i])-int(reference[i]);sum+=d*d;}
    return double(sum)/n;
}
