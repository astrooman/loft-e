#ifndef _H_PAFRB_NETWORK
#define _H_PAFRB_NETWORK

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>

using boost::asio::ip::udp;

#define BUFLEN 8092

class Network
{
    public:
        Network(boost::asio::io_service &ios);
        ~Network(void) {netthread.join(); iothread.join();};
    protected:

    private:
        void receive(void);
        void receive_handler(const boost::system::error_code &error, size_t bytes_transferred, int rport);

        boost::array<unsigned char, BUFLEN> rec_buffer;
        boost::asio::io_service ios_;
        unsigned short count;
        std::thread iothread;
        std::thread netthread;
        std::vector<udp::socket> sockets;
        udp::endpoint sender_endpoint;

};

#endif
