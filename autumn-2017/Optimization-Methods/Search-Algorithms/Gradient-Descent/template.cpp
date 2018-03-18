    #include <iostream>
    #include <iomanip>
    #include <cmath>
 
    struct Point {
        double x, y;
        Point() : x(0.0), y(0.0) {}
        Point( double xx, double yy ) : x(xx), y(yy) {}
        friend Point operator+( const Point &a, const Point &b ) {
            return Point(a.x + b.x, a.y + b.y);
        }
        // and so on...
    };
 
    struct Vector {
        Vector() { x = y = 0.0; }
        Vector( double xx, double yy ) : x(xx), y(yy) {}
        Vector operator+( const Vector &v ) const {
            return Vector(x + v.x, y + v.y);
        }
        Vector operator-() const {
            return Vector(-x, -y);
        }
        Vector operator-( const Vector & v ) const {
            return Vector(x - v.x, y - v.y);
        }
        Vector operator*( double s ) const {
            return Vector(s*x, s*y);
        }
        // ... whatever...
 
        double x, y;
    };
 
    Vector operator*( double s, const Vector &v ) {
        return v * s;
    }
    Point operator+( const Point &a, const Vector &b ) {
        return Point(a.x + b.x, a.y + b.y);
    }
    Point operator-( const Point &a, const Vector &b ) {
        return Point(a.x - b.x, a.y - b.y);
    }
 
    // OP's function
    struct MyFuncVec {
        double operator()( const Point &p ) const {
            //return p.x * p.x * p.x  +  p.y * p.y * p.y  -  2.0 * p.x * p.y;
            return ((p.x - 3.0) * (p.x - 3.0) + 2.0 * (p.y - 4.0) * (p.y - 4.0));
        }
    };
 
 
    struct MyFuncGradient {
        Vector operator()( const Point &p ) {
            //return Vector(3.0 * p.x * p.x  -  2.0 * p.y, 3.0 * p.y * p.y  -  2.0 * p.x);
            return Vector(2.0 * p.x - 6.0, 2.0 * p.y - 8.0);
        }
    };     
 
    struct NumericGradient { 
        NumericGradient( double hh = 0.0001 ) : h(hh), inv_2h(1.0 / (2.0 * hh) ) {}
        template < typename F >
        Vector operator()( F func, Point &p ) {
            double fx0 = func(p.x - h, p.y),
                   fx1 = func(p.x + h, p.y),
                   fy0 = func(p.x, p.y - h),
                   fy1 = func(p.x, p.y + h);
            return Vector((fx1 - fx0) * inv_2h, (fy1 - fy0) * inv_2h);
        }
 
    private:
        double h, inv_2h;
    };
 
    template < typename F >
    double numeric_derivative( F func, double x ) {
        static double h = 0.00001;
        return (func(x + h) - func(x)) / h;
    } 
 
    template < typename F >
    void tabulate_function( F func, double a, double b, int steps ) {
        //  the functor     ^^^^^^  is passed to the templated function
        double step = (b - a) / (steps - 1);
 
        std::cout << "    x          f(x)\n------------------------\n";
        for ( int i = 0; i < steps; ++i ) {
            double x = a + i * step,
                   fx = func(x);
            //          ^^^^^^^ call the operator() of the functor
            std::cout << std::fixed << std::setw(8) << std::setprecision(3) << x
                      << std::scientific << std::setw(16) << std::setprecision(5)
                      << fx << '\n';
        }   
    }

// Monodimensional Optimization with golden ratio search
auto cmpMin = [] ( double f1, double f2 ) -> bool { return f1 < f2; };
auto cmpMax = [] ( double f1, double f2 ) -> bool { return f2 < f1; };
 
class Optimum
{
    public:
        Optimum(double eps = 0.0, unsigned int max = 0)
                : epsilon(setEpsilon(eps)), maxIter(setMaxIter(max)) {}
        
        double setEpsilon(double eps)
        {
            return epsilon = eps > 0.0 ? eps : defaultEpsilon;
        }
            
        int setMaxIter(int max)
        {
            return maxIter = max > 0  &&  max < maxAcceptableIter ? max : defaultMaxIter;
        }
     
        template < typename F >
        double findMin(F func, double a, double b)
        {
            return findOptimum(func, a, b, cmpMin);
        }
     
        template < typename F >
        double findMax(F func, double a, double b)
        {
            return findOptimum(func, a, b, cmpMax);
        }
     
        template <typename F, typename P>
        double findOptimum(F func, double a, double b, P cmp)
        {
            int k = 0;                        
     
            double b_a = b - a;
            double x1  = a + oneTau * b_a; // Computing x values
            double x2  = a + tau * b_a;
     
            double f_x1 = func(x1); // Computing values in x points
            double f_x2 = func(x2);
     
            while (std::fabs(b_a) > epsilon && k < maxIter)
            {
                ++k;
                
                if (cmp(f_x1, f_x2))
                {
                    b    = x2;
                    b_a  = b - a;
                    x2   = x1;
                    x1   = a + oneTau * b_a;
                    f_x2 = f_x1;
                    f_x1 = func(x1);
                }
                else
                {
                    a    = x1;
                    b_a  = b - a;
                    x1   = x2;
                    x2   = a + tau * b_a;
                    f_x1 = f_x2;
                    f_x2 = func(x2);
                }
            }

            std::cout << "Counter: " << k << std::endl;
            // chooses minimum point
            return  cmp(f_x1,f_x2) ? x1 : x2;    
        }
 
    private:
        static constexpr double tau            = (std::sqrt(5.0) - 1.0) / 2.0; // Golden ratio
        static constexpr double oneTau         = 1.0 - tau;
        static constexpr double defaultEpsilon = 1e-3;
        static const int defaultMaxIter        = 100;
        static const int maxAcceptableIter     = 1000;
        double epsilon; // Accuracy value
        int maxIter;
};

void test(double eps)
{
    MyFuncVec funcOP;
    MyFuncGradient gradFuncOP;
    Point p0(0.0, 0.0);
    Vector g = gradFuncOP(p0);
    
    // Use a lambda to transform the OP function to 1D
    auto slicedFunc = [&funcOP, &p0, &g] ( double t ) -> double {
        return funcOP(p0 - t * g);
    };

    Optimum example(eps);

    double t = example.findMin(slicedFunc, 0.0, 1.0);

    Point p1 = p0 - t * g;

    std::cout << "p(x)\tp(y)\t\tf(p)" << std::endl;
    std::cout << p1.x << " " << p1.y << "\t\t" << slicedFunc(t) << std::endl << std::endl;
        
    p0 = p1;
    g = gradFuncOP(p1);    
}

int main(int argc, char const *argv[])
{
    test(1e-1);
    test(1e-3);
    test(1e-6);

    return 0;
}
