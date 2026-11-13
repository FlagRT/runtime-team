// Native vendor XDNN kernel only. Select by PCI attribute before any allocation.
#include <xpu/xdnn.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

static void stage(const char* name) {
  std::cout << "stage " << name << std::endl;
}
static void require_ok(int code,const char* name) {
  std::cout << "return " << name << ' ' << code << std::endl;
  if(code!=0) throw code;
}
int main(int argc,char**argv) {
  if(argc!=2) return 2;
  try {
    stage("enumerate");
    int count=0;
    require_ok(xpu_device_count(&count),"xpu_device_count");
    std::vector<int> ids(count);
    require_ok(xpu_device_list(ids.data()),"xpu_device_list");
    int chosen=-1;
    for(int id:ids) {
      uint64_t address=0;
      require_ok(xpu_device_get_attr(&address,XPUATTR_PCI_ADDRESS,id),"xpu_device_get_attr");
      char pci[32];
      std::snprintf(pci,sizeof(pci),"%04x:%02x:%02x.%x",
        unsigned(address>>16)&65535,unsigned(address>>8)&255,
        unsigned(address>>3)&31,unsigned(address)&7);
      std::cout<<"identity "<<id<<' '<<pci<<std::endl;
      if(std::strcmp(pci,argv[1])==0) {
        if(chosen!=-1) return 3;
        chosen=id;
      }
    }
    if(chosen<0) return 4;
    stage("select_verified_pci");
    require_ok(xpu_set_device(chosen),"xpu_set_device");
    int current=-1;
    require_ok(xpu_current_device(&current),"xpu_current_device");
    if(current!=chosen) return 5;
    stage("create_xdnn_context");
    auto*ctx=baidu::xpu::api::create_context();
    if(!ctx) return 6;
    float*device=nullptr;
    stage("allocate");
    require_ok(xpu_malloc(reinterpret_cast<void**>(&device),4*sizeof(float)),"xpu_malloc");
    stage("constant_kernel");
    require_ok(baidu::xpu::api::constant<float>(ctx,device,4,1.0f),"xdnn_constant");
    stage("wait_for_kernel");
    require_ok(xpu_wait(),"xpu_wait");
    stage("readback");
    float host[4]={};
    require_ok(xpu_memcpy(host,device,sizeof(host),XPU_DEVICE_TO_HOST),"xpu_memcpy");
    for(float value:host) if(value!=1.0f) return 7;
    require_ok(xpu_free(device),"xpu_free");
    baidu::xpu::api::destroy_context(ctx);
    std::cout<<"PASS native_xdnn_constant 4/4"<<std::endl;
    return 0;
  } catch(int error) { return error?error:8; }
}
