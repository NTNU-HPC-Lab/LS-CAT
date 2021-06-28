

class Renderer
{
public:
    Renderer();
    ~Renderer();

    void renderImage( int width, int height );

private:
    void writeImageToDisk( std::string filename, int width, int height, float* pixelData );
};

