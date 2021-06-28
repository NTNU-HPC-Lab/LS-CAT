class MyClass {
public:
    double hostParam;

    MyClass(){

    }

    ~MyClass(){

    }

    void set_param(double in){
        hostParam = in;
    }

    double do_it_on_host(){
        double out;
        hostKernel(&hostParam, &out);
        return out;
    }

    void hostKernel(double *param, double *ans){
        // Host implementation
        std::cout << "Inside hostKernel: " << "wow" << std::endl;
        *ans = *param + 3.14;
    }
};