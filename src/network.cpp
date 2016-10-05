#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>

#include "network.hpp"
#include "pdih.hpp"

using boost::asio::ip::udp;
using std::cout;
using std::endl;



Network::Network(boost::asio::io_servie &ios, Pool &mypool) : count(0), highest_frame(-1)
{
    boost::asio::socket_base::reuse_address option1(true);
    boost::asio::socket_base::receive_buffer_size option2(8000);

    for (ii = 0; ii < 6; ii++) {
        sockets.push_back(udp::socket(ios, udp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 17100 + ii)));
        sockets[ii].set_option(option1);
        sockets[ii].set_option(option2);
    }

    cout << "Waiting to receive something..." << endl;
    cout.flush();

    netthread = std::thread(&Network::receive, this, mypool);
    // sleeping makes sure that async_receive_from() was called before ios.run() call
    std::this_thread::sleep_for(std::chrono::seconds(1));   // need to sort this out properly later
    iothread = std::thread([&ios]{ios.run();});
}

void Network::receive(Pool &mypool)
{
    sockets[0].async_receive_from(boost::asio::buffer(rec_buffer), sender_endpoint, boost::bind(&Network::receive_handler, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred, 17100, mypool));
}

void Network::receive_handler(const boost::system::error_code &error, size_t bytes_transferred, int rport, Pool &mypool)
{
    // there will be so many race conditions here when we call it with multiple threads
    count++;
    header_s head;
    // possible race conditions here
    static obs_time start_time{head.epoch, head.ref_s};
    get_header(rec_buffer.data(), head);
    mypool.get_data(rec_buffer.data(), head.frame_no + (head.ref_s - start_time.start_second) * 250000, highest_frame, head.thread, start_time);
    if (count < 1024)
        receive();
}
