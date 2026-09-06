public struct Complex<let n : uint>
{
    public vector<float, n> real;
    public vector<float, n> imaginary;

    public __init(vector<float, n> r, vector<float, n> i)
    {
        real = r;
        imaginary = i;
    }

    public __init(vector<float, n> r)
    {
        real = r;
        imaginary = 0;
    }

    public __init(float r)
    {
        real = r;
        imaginary = 0;
    }
}

public typedef Complex<4> complex4;

public Complex<n> operator+<let n : uint>(Complex<n> a, Complex<n> b)
{
    return Complex<n>(a.real + b.real, a.imaginary + b.imaginary);
}

public Complex<n> operator-<let n : uint>(Complex<n> a, Complex<n> b)
{
    return Complex<n>(a.real - b.real, a.imaginary - b.imaginary);
}

public Complex<n> operator*<let n : uint>(Complex<n> a, Complex<n> b)
{
    return Complex<n>(
        a.real * b.real - a.imaginary * b.imaginary,
        a.real * b.imaginary + a.imaginary * b.real);
}

public Complex<n> operator/<let n : uint>(Complex<n> a, Complex<n> b)
{
    vector<float, n> scale = 1.0 / (b.real * b.real + b.imaginary * b.imaginary);
    return Complex<n>(
        (a.real * b.real + a.imaginary * b.imaginary) * scale,
        (a.imaginary * b.real - a.real * b.imaginary) * scale);
}

public Complex<n> operator+<let n : uint>(Complex<n> a, float b) { return a + Complex<n>(b); }
public Complex<n> operator-<let n : uint>(Complex<n> a, float b) { return a - Complex<n>(b); }
public Complex<n> operator*<let n : uint>(Complex<n> a, float b) { return a * Complex<n>(b); }
public Complex<n> operator/<let n : uint>(Complex<n> a, float b) { return a / Complex<n>(b); }

public Complex<n> operator+<let n : uint>(float a, Complex<n> b) { return Complex<n>(a) + b; }
public Complex<n> operator-<let n : uint>(float a, Complex<n> b) { return Complex<n>(a) - b; }
public Complex<n> operator*<let n : uint>(float a, Complex<n> b) { return Complex<n>(a) * b; }
public Complex<n> operator/<let n : uint>(float a, Complex<n> b) { return Complex<n>(a) / b; }

public Complex<n> operator+<let n : uint>(Complex<n> a, vector<float, n> b) { return a + Complex<n>(b); }
public Complex<n> operator-<let n : uint>(Complex<n> a, vector<float, n> b) { return a - Complex<n>(b); }
public Complex<n> operator*<let n : uint>(Complex<n> a, vector<float, n> b) { return a * Complex<n>(b); }
public Complex<n> operator/<let n : uint>(Complex<n> a, vector<float, n> b) { return a / Complex<n>(b); }

public Complex<n> operator+<let n : uint>(vector<float, n> a, Complex<n> b) { return Complex<n>(a) + b; }
public Complex<n> operator-<let n : uint>(vector<float, n> a, Complex<n> b) { return Complex<n>(a) - b; }
public Complex<n> operator*<let n : uint>(vector<float, n> a, Complex<n> b) { return Complex<n>(a) * b; }
public Complex<n> operator/<let n : uint>(vector<float, n> a, Complex<n> b) { return Complex<n>(a) / b; }

public vector<float, n> norm<let n : uint>(Complex<n> z)
{
    return z.real * z.real + z.imaginary * z.imaginary;
}

public vector<float, n> abs<let n : uint>(Complex<n> z)
{
    return sqrt(norm(z));
}

// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/pstd.h#L1218 more precise version, vectorized
public Complex<n> sqrt<let n : uint>(Complex<n> z)
{
    vector<float, n> magnitude = abs(z);
    vector<float, n> t1 = sqrt(0.5 * (magnitude + abs(z.real)));
    vector<float, n> t2 = 0.5 * z.imaginary / max(t1, 1e-30);
    vector<float, n> signed_t1 = select(asint(z.imaginary) < 0, -t1, t1);

    vector<float, n> real = select(z.real >= 0.0, t1, abs(t2));
    vector<float, n> imaginary = select(z.real >= 0.0, t2, signed_t1);

    return Complex<n>(
        select(magnitude == 0.0, 0.0, real),
        select(magnitude == 0.0, 0.0, imaginary));
}
