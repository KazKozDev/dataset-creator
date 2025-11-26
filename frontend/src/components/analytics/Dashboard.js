import React from 'react';
import {
    Box,
    SimpleGrid,
    Stat,
    StatLabel,
    StatNumber,
    StatHelpText,
    StatArrow,
    Heading,
    Card,
    CardBody,
} from '@chakra-ui/react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    BarChart,
    Bar
} from 'recharts';

const data = [
    { name: 'Mon', cost: 4000, quality: 2400, amt: 2400 },
    { name: 'Tue', cost: 3000, quality: 1398, amt: 2210 },
    { name: 'Wed', cost: 2000, quality: 9800, amt: 2290 },
    { name: 'Thu', cost: 2780, quality: 3908, amt: 2000 },
    { name: 'Fri', cost: 1890, quality: 4800, amt: 2181 },
    { name: 'Sat', cost: 2390, quality: 3800, amt: 2500 },
    { name: 'Sun', cost: 3490, quality: 4300, amt: 2100 },
];

const Dashboard = () => {
    return (
        <Box p={5}>
            <Heading size="md" mb={5}>Analytics Dashboard</Heading>

            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={5} mb={10}>
                <Card>
                    <CardBody>
                        <Stat>
                            <StatLabel>Total Generated</StatLabel>
                            <StatNumber>1,234</StatNumber>
                            <StatHelpText>
                                <StatArrow type='increase' />
                                23.36%
                            </StatHelpText>
                        </Stat>
                    </CardBody>
                </Card>
                <Card>
                    <CardBody>
                        <Stat>
                            <StatLabel>Avg Quality Score</StatLabel>
                            <StatNumber>8.5</StatNumber>
                            <StatHelpText>
                                <StatArrow type='increase' />
                                5.05%
                            </StatHelpText>
                        </Stat>
                    </CardBody>
                </Card>
                <Card>
                    <CardBody>
                        <Stat>
                            <StatLabel>Total Cost</StatLabel>
                            <StatNumber>$45.00</StatNumber>
                            <StatHelpText>
                                <StatArrow type='decrease' />
                                9.05%
                            </StatHelpText>
                        </Stat>
                    </CardBody>
                </Card>
            </SimpleGrid>

            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={10}>
                <Box h="400px" bg="white" p={4} borderRadius="lg" boxShadow="sm">
                    <Heading size="md" mb={4}>Generation Trends</Heading>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="quality" stroke="#8884d8" activeDot={{ r: 8 }} />
                            <Line type="monotone" dataKey="cost" stroke="#82ca9d" />
                        </LineChart>
                    </ResponsiveContainer>
                </Box>

                <Box h="400px" bg="white" p={4} borderRadius="lg" boxShadow="sm">
                    <Heading size="md" mb={4}>Cost Analysis</Heading>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="cost" fill="#8884d8" />
                            <Bar dataKey="amt" fill="#82ca9d" />
                        </BarChart>
                    </ResponsiveContainer>
                </Box>
            </SimpleGrid>
        </Box>
    );
};

export default Dashboard;
