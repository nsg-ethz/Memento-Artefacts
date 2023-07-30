/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

// Network topology
//
//                  disturbance
//                      |
//        sender --- switchA --- switchB --- receiver
//        sender --/
//        sender --/
//        (...)
//
//  The "disturbance" host is used to introduce changes in the network
//  conditions.
//

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/bridge-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

#include "ns3/cdf-application.h"
#include "ns3/experiment-tags.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TrafficGenerationExperiment");

const auto TCP = TypeIdValue(TcpSocketFactory::GetTypeId());
const auto UDP = TypeIdValue(UdpSocketFactory::GetTypeId());

// Put the current timestamp and packet size into a log.
void logSize(Ptr<OutputStreamWrapper> stream, Ptr<Packet const> p)
{
    auto current_time = Simulator::Now();
    *stream->GetStream() << current_time.GetSeconds() << ','
                         << p->GetSize() << std::endl;
}

// Tag a packet with a timestamp.
void setTimeTag(Ptr<Packet const> p)
{
    TimestampTag tag;
    tag.SetTime(Simulator::Now());
    p->AddPacketTag(tag);
};

void setIdTag(u_int32_t workload_id, u_int32_t app_id, Ptr<Packet const> p)
{
    IdTag tag;
    tag.SetWorkload(workload_id);
    tag.SetApplication(app_id);
    p->AddPacketTag(tag);
};

// Log workload tag, timestamp tag, and packet size.
void logPacketInfo(Ptr<OutputStreamWrapper> stream, Ptr<Packet const> p)
{
    TimestampTag timestampTag;
    IdTag idTag;
    if (p->PeekPacketTag(timestampTag) && p->PeekPacketTag(idTag))
    {
        auto current_time = Simulator::Now();
        auto diff_time = current_time - timestampTag.GetTime();
        *stream->GetStream() << current_time.GetSeconds() << ','
                             << diff_time.GetSeconds() << ','
                             << p->GetSize() << ','
                             << idTag.GetWorkload() << ','
                             << idTag.GetApplication() << std::endl;
    }
    else
    {
        // NS_LOG_DEBUG("Packet without timestamp, won't log.");
    };
};

void logValue(Ptr<OutputStreamWrapper> stream, std::string context,
              uint32_t oldval, uint32_t newval)
{
    auto current_time = Simulator::Now();
    *stream->GetStream() << context << ',' << current_time.GetSeconds() << ','
                         << newval << std::endl;
}

void logDrop(Ptr<OutputStreamWrapper> stream,
             std::string context, Ptr<Packet const> p)
{
    auto current_time = Simulator::Now();
    *stream->GetStream() << context << ',' << current_time.GetSeconds() << ','
                         << p->GetSize() << std::endl;
};

// TODO: Add base stream? Or how to get different random streams?
Ptr<RandomVariableStream> TimeStream(double min = 0.0, double max = 1.0)
{
    return CreateObjectWithAttributes<UniformRandomVariable>(
        "Min", DoubleValue(min),
        "Max", DoubleValue(max));
}

PointerValue TimeStreamValue(double min = 0.0, double max = 1.0)
{
    return PointerValue(TimeStream(min, max));
}

NetDeviceContainer GetNetDevices(Ptr<Node> node)
{
    NetDeviceContainer devices;
    for (uint32_t i = 0; i < node->GetNDevices(); ++i)
    {
        devices.Add(node->GetDevice(i));
    }
    return devices;
}

int main(int argc, char *argv[])
{
    //
    // Users may find it convenient to turn on explicit debugging
    // for selected modules; the below lines suggest how to do this
    //
#if 1
    LogComponentEnable("TrafficGenerationExperiment", LOG_LEVEL_INFO);
#endif
    // Simulation variables
    auto simStart = TimeValue(Seconds(0));
    auto stopTime = Seconds(60);
    auto simStop = TimeValue(stopTime);

    //
    // Allow the user to override any of the defaults and the above Bind() at
    // run-time, via command-line arguments
    //

    int n_apps = 20;
    int start_window = 60;
    DataRate linkrate("100Mbps");
    DataRate baserate("20Mbps");
    DataRate congestion("0Mbps");
    Time delay("5ms");
    QueueSize queuesize("100p");

    std::string basedir = "./distributions/";
    std::string w1 = basedir + "Facebook_WebServerDist_IntraCluster.txt";
    std::string w2 = basedir + "DCTCP_MsgSizeDist.txt";
    std::string w3 = basedir + "Facebook_HadoopDist_All.txt";
    std::string prefix = "shift";
    double c_w1 = 1.0 / 3;
    double c_w2 = 1.0 / 3;
    double c_w3 = 1.0 / 3;

    CommandLine cmd;
    cmd.AddValue("apps", "Number of traffic apps per workload.", n_apps);
    cmd.AddValue("startwindow", "Maximum diff in start time.", start_window);
    cmd.AddValue("apprate", "Total traffic rate of all apps.", baserate);
    cmd.AddValue("linkrate", "Link capacity rate.", linkrate);
    cmd.AddValue("linkdelay", "Link delay.", delay);
    cmd.AddValue("queuesize", "Bottleneck queue size.", queuesize);
    cmd.AddValue("w1", "Factor for W1 traffic (FB webserver).", c_w1);
    cmd.AddValue("w2", "Factor for W2 traffic (DCTCP messages).", c_w2);
    cmd.AddValue("w3", "Factor for W3 traffic (FB hadoop).", c_w3);
    cmd.AddValue("congestion", "Congestion traffic rate.", congestion);
    cmd.AddValue("prefix", "Prefix for log files.", prefix);
    cmd.Parse(argc, argv);

    // Compute resulting workload datarates.
    auto appStart = 1.0;
    auto appStop = std::min(appStart + start_window, stopTime.GetSeconds());
    // Spreading out starting time reduces effective rate.
    auto rate_reduction = 0.5 * (appStop - appStart) / (stopTime.GetSeconds() - appStart);
    auto rate_scale = 1 / (1 - rate_reduction);

    NS_LOG_INFO("Starting traffic between "
                << appStart << "s and " << appStop << "s.");
    NS_LOG_INFO("Rate scaling: " << rate_scale);



    auto apprate = baserate.GetBitRate() / n_apps;
    auto rate_w1 = DataRate(static_cast<uint64_t>(rate_scale * c_w1 * apprate));
    auto rate_w2 = DataRate(static_cast<uint64_t>(rate_scale * c_w2 * apprate));
    auto rate_w3 = DataRate(static_cast<uint64_t>(rate_scale * c_w3 * apprate));
    congestion = DataRate(static_cast<uint64_t>(
        rate_scale * congestion.GetBitRate()));

    // Print Overview of seetings
    NS_LOG_DEBUG("Overview:" << std::endl
                 << "Total sender rate: " << baserate << std::endl
                 << "Apps: " << n_apps << std::endl
                 << "Starting traffic between "
                 << appStart << "s and " << appStop << "s." << std::endl
                 << "Start window rate scaling: " << rate_scale << std::endl
                 << "Congestion: " << congestion << std::endl
                 << "Workloads (data rates per app):" << std::endl
                 << "W1: " << w1 << " (" << rate_w1 << ")" << std::endl
                 << "W2: " << w2 << " (" << rate_w2 << ")" << std::endl
                 << "W3: " << w3 << " (" << rate_w3 << ")");

    // Fix MTU and Segment size, otherwise the small TCP default (536) is used.
    Config::SetDefault("ns3::CsmaNetDevice::Mtu", UintegerValue(1500));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1446)); // MTU 1446

    //
    // Explicitly create the nodes required by the topology (shown above).
    //
    NS_LOG_INFO("Create nodes.");
    NodeContainer switches;
    switches.Create(2);
    auto switchA = switches.Get(0);
    auto switchB = switches.Get(1);

    NodeContainer senders;
    senders.Create(3 * n_apps);
    // receiver, and disturbance
    NodeContainer hosts;
    hosts.Create(2);
    auto receiver = hosts.Get(0);
    auto disturbance = hosts.Get(1);

    NS_LOG_INFO("Build Topology");
    CsmaHelper csma;
    csma.SetChannelAttribute("FullDuplex", BooleanValue(true));
    csma.SetChannelAttribute("DataRate", DataRateValue(linkrate));
    csma.SetChannelAttribute("Delay", TimeValue(delay));

    // Create the csma links
    csma.Install(NodeContainer(switchA, switchB));
    csma.Install(NodeContainer(receiver, switchB));
    csma.Install(NodeContainer(disturbance, switchA));

    for (auto it = senders.Begin(); it != senders.End(); it++)
    {
        Ptr<Node> sender = *it;
        csma.Install(NodeContainer(sender, switchA));
    }

    // Update queuesizes
    Config::Set(
        "/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/TxQueue/MaxSize",
        QueueSizeValue(queuesize));

    // Create the bridge netdevice, turning the nodes into actual switches
    BridgeHelper bridge;
    bridge.Install(switchA, GetNetDevices(switchA));
    bridge.Install(switchB, GetNetDevices(switchB));

    // Add internet stack and IP addresses to the hosts
    NS_LOG_INFO("Setup stack and assign IP Addresses.");
    NetDeviceContainer hostDevices;
    hostDevices.Add(GetNetDevices(receiver)); // Needs to be first to get addr.
    for (auto it = senders.Begin(); it != senders.End(); it++)
    {
        Ptr<Node> sender = *it;
        hostDevices.Add(GetNetDevices(sender));
    }
    hostDevices.Add(GetNetDevices(disturbance));

    InternetStackHelper internet;
    internet.Install(senders);
    internet.Install(hosts);
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.0.0", "255.255.0.0");
    auto addresses = ipv4.Assign(hostDevices);
    // Get Address: Device with index 0, address 0 (only one address)
    auto addrReceiver = addresses.GetAddress(0, 0);

    NS_LOG_INFO("Create Traffic Applications.");
    uint16_t base_port = 4200; // Note: We need two ports per pair
    auto trafficStart = TimeStream(appStart, appStop);
    for (auto i_app = 0; i_app < n_apps; ++i_app)
    {
        // We also need to set the appropriate tag at every application!

        // Addresses
        auto port = base_port + i_app;
        auto recvAddr = AddressValue(InetSocketAddress(addrReceiver, port));

        // Sink
        Ptr<Application> sink = CreateObjectWithAttributes<PacketSink>(
            "Local", recvAddr, "Protocol", TCP,
            "StartTime", simStart, "StopTime", simStop);
        receiver->AddApplication(sink);

        // Sources for each workload
        // App indexing scheme: 0--n_apps-1: w1, n_apps -- 2n_apps-1: w2, etc.
        if (rate_w1 > 0)
        {
            auto _id = i_app;
            Ptr<CdfApplication> source1 = CreateObjectWithAttributes<CdfApplication>(
                "Remote", recvAddr, "Protocol", TCP,
                "DataRate", DataRateValue(rate_w1), "CdfFile", StringValue(w1),
                "StartTime", TimeValue(Seconds(trafficStart->GetValue())),
                "StopTime", simStop);
            source1->TraceConnectWithoutContext(
                "Tx", MakeBoundCallback(&setIdTag, 1, _id));
            senders.Get(_id)->AddApplication(source1);
        }
        if (rate_w2 > 0)
        {
            auto _id = i_app + n_apps;
            Ptr<CdfApplication> source2 = CreateObjectWithAttributes<CdfApplication>(
                "Remote", recvAddr, "Protocol", TCP,
                "DataRate", DataRateValue(rate_w2), "CdfFile", StringValue(w2),
                "StartTime", TimeValue(Seconds(trafficStart->GetValue())),
                "StopTime", simStop);
            source2->TraceConnectWithoutContext(
                "Tx", MakeBoundCallback(&setIdTag, 2, _id));
            senders.Get(_id)->AddApplication(source2);
        }
        if (rate_w3 > 0)
        {
            auto _id = i_app + (2 * n_apps);
            Ptr<CdfApplication> source3 = CreateObjectWithAttributes<CdfApplication>(
                "Remote", recvAddr, "Protocol", TCP,
                "DataRate", DataRateValue(rate_w3), "CdfFile", StringValue(w3),
                "StartTime", TimeValue(Seconds(trafficStart->GetValue())),
                "StopTime", simStop);
            source3->TraceConnectWithoutContext(
                "Tx", MakeBoundCallback(&setIdTag, 3, _id));
            senders.Get(_id)->AddApplication(source3);
        }
    }

    if (congestion > 0)
    {
        NS_LOG_INFO("Configure congestion app.");
        // Just blast UDP traffic from time to time
        Ptr<Application> congestion_sink = CreateObjectWithAttributes<PacketSink>(
            "Local", AddressValue(InetSocketAddress(addrReceiver, 2100)),
            "Protocol", UDP, "StartTime", simStart, "StopTime", simStop);
        receiver->AddApplication(congestion_sink);

        Ptr<Application> congestion_source = CreateObjectWithAttributes<OnOffApplication>(
            "Remote", AddressValue(InetSocketAddress(addrReceiver, 2100)),
            "Protocol", UDP,
            "OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"),
            "OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"),
            "DataRate", DataRateValue(congestion),
            "StartTime", TimeValue(Seconds(appStart)),
            "StopTime", simStop);
        disturbance->AddApplication(congestion_source);
    }
    else
    {
        NS_LOG_INFO("No congestion.");
    }

    NS_LOG_INFO("Install Tracing");
    AsciiTraceHelper asciiTraceHelper;

    // Log (one-way) delay from sender to receiver (excludes other sources).
    std::stringstream trackfilename;
    // trackfilename << prefix << "_delays.csv";
    trackfilename << prefix << ".csv"; // only one file for now.
    auto trackfile = asciiTraceHelper.CreateFileStream(trackfilename.str());
    for (auto it = senders.Begin(); it != senders.End(); it++)
    {
        Ptr<Node> sender = *it;
        sender->GetDevice(0)->TraceConnectWithoutContext(
            "MacTx", MakeCallback(&setTimeTag));
    }
    receiver->GetDevice(0)->TraceConnectWithoutContext(
        "MacRx", MakeBoundCallback(&logPacketInfo, trackfile));

    // Track queues
    // std::stringstream queuefilename;
    // queuefilename << prefix << "_queue.csv";
    // auto queuefile = asciiTraceHelper.CreateFileStream(queuefilename.str());
    // Config::Connect(
    //     "/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/TxQueue/PacketsInQueue",
    //     MakeBoundCallback(&logValue, queuefile));

    // Track drops
    // auto dropfile = asciiTraceHelper.CreateFileStream("drops.csv");
    // Config::Connect(
    //     "/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/MacTxDrop",
    //     MakeBoundCallback(&logDrop, dropfile));

    // csma.EnablePcapAll("csma-bridge", false);

    //
    // Now, do the actual simulation.
    //
    NS_LOG_INFO("Run Simulation.");
    Simulator::Stop(stopTime);
    Simulator::Run();
    Simulator::Destroy();
    NS_LOG_INFO("Done.");
}
