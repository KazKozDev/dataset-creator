import React from 'react';
import {
  Box,
  Heading,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Card,
  CardBody,
  CardHeader,
  Text,
  Flex,
  Badge,
  Icon,
  Divider,
  Button,
  useColorModeValue,
} from '@chakra-ui/react';
import { FiDatabase, FiCpu, FiCheckCircle, FiActivity, FiArrowRight, FiServer } from 'react-icons/fi';
import { useQuery } from '@tanstack/react-query';
import { getDatasets, getAnalyticsSummary, getAllTasks } from '../services/api';
import { Link } from 'react-router-dom';
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
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const Dashboard = () => {
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Fetch datasets
  const { data: datasetsData, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => getDatasets()
  });

  // Fetch analytics data
  const { data: analyticsData } = useQuery({
    queryKey: ['analytics', 7], // Last 7 days
    queryFn: () => getAnalyticsSummary(7)
  });

  // Fetch tasks data
  const { data: tasksData } = useQuery({
    queryKey: ['tasks'],
    queryFn: getAllTasks,
    refetchInterval: 5000
  });

  // Calculate summary statistics
  const calculateStats = () => {
    if (!datasetsData?.datasets) return { total: 0, formats: {}, domains: {} };

    const datasets = datasetsData.datasets;
    const formats = {};
    const domains = {};

    datasets.forEach(dataset => {
      // Count formats
      formats[dataset.format] = (formats[dataset.format] || 0) + 1;

      // Count domains
      domains[dataset.domain] = (domains[dataset.domain] || 0) + 1;
    });

    return {
      total: datasets.length,
      formats,
      domains
    };
  };

  const stats = calculateStats();

  // Sample of recent datasets
  const recentDatasets = datasetsData?.datasets?.slice(0, 5) || [];

  // Prepare chart data from analytics
  const prepareChartData = () => {
    if (!analyticsData?.daily_stats) {
      // Fallback to empty data
      return [];
    }

    return analyticsData.daily_stats.map(stat => ({
      name: new Date(stat.date).toLocaleDateString('en-US', { weekday: 'short' }),
      generated: stat.examples_generated || 0,
      quality: stat.average_quality || 0,
      cost: stat.total_cost || 0
    })).slice(-7); // Show only last 7 days
  };

  const chartData = prepareChartData();

  // Calculate task stats
  const taskStats = () => {
    const tasks = tasksData?.tasks || [];
    const running = tasks.filter(t => t.status === 'running').length;
    const pending = tasks.filter(t => t.status === 'pending').length;
    const completed = tasks.filter(t => t.status === 'completed').length;
    const failed = tasks.filter(t => t.status === 'failed').length;
    return { running, pending, completed, failed, total: tasks.length };
  };

  const tasks = taskStats();

  // Calculate total examples across all datasets
  const totalExamples = datasetsData?.datasets?.reduce((sum, d) => sum + (d.example_count || 0), 0) || 0;

  // Get generation stats from analytics
  const generationStats = analyticsData?.generation || {};
  const qualityStats = analyticsData?.quality || {};

  // Prepare models/providers data for charts
  const COLORS = ['#3182CE', '#38A169', '#805AD5', '#DD6B20', '#E53E3E', '#00B5D8', '#D69E2E'];
  
  const modelsData = Object.entries(generationStats.models || {})
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 5);
  
  const providersData = Object.entries(generationStats.providers || {})
    .map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }))
    .sort((a, b) => b.value - a.value);

  return (
    <Box>
      <Heading size="md" mb={6}>Dashboard</Heading>

      {/* Stats cards with real data */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        {/* Generation Stats */}
        <Card
          as={Link}
          to="/generator"
          bg={cardBg}
          borderWidth="1px"
          borderRadius="lg"
          borderColor={borderColor}
          boxShadow="md"
          transition="all 0.2s"
          _hover={{ transform: 'translateY(-4px)', boxShadow: 'lg', textDecoration: 'none' }}
          cursor="pointer"
        >
          <CardBody>
            <Flex align="center" mb={2}>
              <Icon as={FiCpu} fontSize="xl" color="blue.500" mr={2} />
              <Text fontWeight="semibold" color="gray.600">Generation</Text>
            </Flex>
            <Stat>
              <StatNumber fontSize="2xl">{generationStats.total_examples || 0}</StatNumber>
              <StatHelpText mb={0}>
                examples generated
              </StatHelpText>
            </Stat>
            <Flex mt={2} gap={2} flexWrap="wrap">
              <Badge colorScheme="blue">{generationStats.total_jobs || 0} jobs</Badge>
              {generationStats.total_cost_usd > 0 && (
                <Badge colorScheme="green">${generationStats.total_cost_usd}</Badge>
              )}
            </Flex>
          </CardBody>
        </Card>

        {/* Quality Stats */}
        <Card
          as={Link}
          to="/quality"
          bg={cardBg}
          borderWidth="1px"
          borderRadius="lg"
          borderColor={borderColor}
          boxShadow="md"
          transition="all 0.2s"
          _hover={{ transform: 'translateY(-4px)', boxShadow: 'lg', textDecoration: 'none' }}
          cursor="pointer"
        >
          <CardBody>
            <Flex align="center" mb={2}>
              <Icon as={FiCheckCircle} fontSize="xl" color="green.500" mr={2} />
              <Text fontWeight="semibold" color="gray.600">Quality</Text>
            </Flex>
            <Stat>
              <StatNumber fontSize="2xl">
                {qualityStats.average_quality_score || '—'}
                {qualityStats.average_quality_score && <Text as="span" fontSize="md" color="gray.500">/10</Text>}
              </StatNumber>
              <StatHelpText mb={0}>
                avg quality score
              </StatHelpText>
            </Stat>
            <Flex mt={2} gap={2} flexWrap="wrap">
              <Badge colorScheme="green">{qualityStats.total_checks || 0} checks</Badge>
              {qualityStats.total_issues_found > 0 && (
                <Badge colorScheme="orange">{qualityStats.total_issues_found} issues</Badge>
              )}
            </Flex>
          </CardBody>
        </Card>

        {/* Datasets Stats */}
        <Card
          as={Link}
          to="/datasets"
          bg={cardBg}
          borderWidth="1px"
          borderRadius="lg"
          borderColor={borderColor}
          boxShadow="md"
          transition="all 0.2s"
          _hover={{ transform: 'translateY(-4px)', boxShadow: 'lg', textDecoration: 'none' }}
          cursor="pointer"
        >
          <CardBody>
            <Flex align="center" mb={2}>
              <Icon as={FiDatabase} fontSize="xl" color="purple.500" mr={2} />
              <Text fontWeight="semibold" color="gray.600">Datasets</Text>
            </Flex>
            <Stat>
              <StatNumber fontSize="2xl">{stats.total}</StatNumber>
              <StatHelpText mb={0}>
                {totalExamples.toLocaleString()} total examples
              </StatHelpText>
            </Stat>
            <Flex mt={2} gap={2} flexWrap="wrap">
              {Object.entries(stats.formats).slice(0, 2).map(([format, count]) => (
                <Badge key={format} colorScheme={format === 'chat' ? 'blue' : 'green'}>
                  {format}: {count}
                </Badge>
              ))}
            </Flex>
          </CardBody>
        </Card>

        {/* Tasks Stats */}
        <Card
          as={Link}
          to="/tasks"
          bg={cardBg}
          borderWidth="1px"
          borderRadius="lg"
          borderColor={borderColor}
          boxShadow="md"
          transition="all 0.2s"
          _hover={{ transform: 'translateY(-4px)', boxShadow: 'lg', textDecoration: 'none' }}
          cursor="pointer"
        >
          <CardBody>
            <Flex align="center" mb={2}>
              <Icon as={FiActivity} fontSize="xl" color="orange.500" mr={2} />
              <Text fontWeight="semibold" color="gray.600">Tasks</Text>
            </Flex>
            <Stat>
              <StatNumber fontSize="2xl">
                {tasks.running > 0 ? (
                  <Flex align="center" gap={2}>
                    {tasks.running}
                    <Badge colorScheme="blue" fontSize="xs" animation="pulse 2s infinite">
                      running
                    </Badge>
                  </Flex>
                ) : tasks.pending > 0 ? (
                  <Flex align="center" gap={2}>
                    {tasks.pending}
                    <Badge colorScheme="yellow" fontSize="xs">pending</Badge>
                  </Flex>
                ) : (
                  tasks.total
                )}
              </StatNumber>
              <StatHelpText mb={0}>
                {tasks.running > 0 || tasks.pending > 0 ? 'active tasks' : 'total tasks'}
              </StatHelpText>
            </Stat>
            <Flex mt={2} gap={2} flexWrap="wrap">
              <Badge colorScheme="green">{tasks.completed} done</Badge>
              {tasks.failed > 0 && (
                <Badge colorScheme="red">{tasks.failed} failed</Badge>
              )}
            </Flex>
          </CardBody>
        </Card>
      </SimpleGrid>

      {/* Analytics Section */}
      <Heading size="sm" mb={4} mt={6}>Analytics</Heading>
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6} mb={8}>
        <Card bg={cardBg} borderWidth="1px" borderRadius="lg" borderColor={borderColor}>
          <CardHeader>
            <Heading size="sm">Generation Trends</Heading>
          </CardHeader>
          <Divider />
          <CardBody>
            <Box h="300px">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 10]} />
                  <Tooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="generated" stroke="#3182CE" name="Generated" />
                  <Line yAxisId="right" type="monotone" dataKey="quality" stroke="#38A169" name="Avg Quality" />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </CardBody>
        </Card>

        <Card bg={cardBg} borderWidth="1px" borderRadius="lg" borderColor={borderColor}>
          <CardHeader>
            <Heading size="sm">Cost Analysis</Heading>
          </CardHeader>
          <Divider />
          <CardBody>
            <Box h="300px">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="cost" fill="#805AD5" name="Cost ($)" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </CardBody>
        </Card>
      </SimpleGrid>

      {/* Models & Providers Section */}
      {(modelsData.length > 0 || providersData.length > 0) && (
        <>
          <Heading size="sm" mb={4} mt={6}>
            <Flex align="center" gap={2}>
              <Icon as={FiServer} />
              Models & Providers
            </Flex>
          </Heading>
          <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6} mb={8}>
            {/* Top Models */}
            <Card bg={cardBg} borderWidth="1px" borderRadius="lg" borderColor={borderColor}>
              <CardHeader>
                <Heading size="sm">Top Models Used</Heading>
              </CardHeader>
              <Divider />
              <CardBody>
                {modelsData.length > 0 ? (
                  <Flex direction={{ base: 'column', md: 'row' }} align="center" gap={4}>
                    <Box h="200px" w={{ base: '100%', md: '50%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={modelsData}
                            cx="50%"
                            cy="50%"
                            innerRadius={40}
                            outerRadius={80}
                            paddingAngle={2}
                            dataKey="value"
                          >
                            {modelsData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                    <Box flex={1}>
                      {modelsData.map((model, index) => (
                        <Flex key={model.name} align="center" mb={2} justify="space-between">
                          <Flex align="center" gap={2}>
                            <Box w={3} h={3} borderRadius="sm" bg={COLORS[index % COLORS.length]} />
                            <Text fontSize="sm" noOfLines={1} maxW="150px" title={model.name}>
                              {model.name}
                            </Text>
                          </Flex>
                          <Badge colorScheme="gray">{model.value} jobs</Badge>
                        </Flex>
                      ))}
                    </Box>
                  </Flex>
                ) : (
                  <Text color="gray.500" textAlign="center" py={8}>
                    No model usage data yet. Generate some datasets!
                  </Text>
                )}
              </CardBody>
            </Card>

            {/* Providers Distribution */}
            <Card bg={cardBg} borderWidth="1px" borderRadius="lg" borderColor={borderColor}>
              <CardHeader>
                <Heading size="sm">Providers Distribution</Heading>
              </CardHeader>
              <Divider />
              <CardBody>
                {providersData.length > 0 ? (
                  <Box>
                    {providersData.map((provider, index) => {
                      const total = providersData.reduce((sum, p) => sum + p.value, 0);
                      const percentage = Math.round((provider.value / total) * 100);
                      return (
                        <Box key={provider.name} mb={4}>
                          <Flex justify="space-between" mb={1}>
                            <Text fontWeight="medium">{provider.name}</Text>
                            <Text color="gray.600">{provider.value} jobs ({percentage}%)</Text>
                          </Flex>
                          <Box bg="gray.100" borderRadius="full" h={2}>
                            <Box
                              bg={COLORS[index % COLORS.length]}
                              h={2}
                              borderRadius="full"
                              w={`${percentage}%`}
                              transition="width 0.3s"
                            />
                          </Box>
                        </Box>
                      );
                    })}
                    <Divider my={4} />
                    <Flex justify="space-between" color="gray.600" fontSize="sm">
                      <Text>Total API Calls</Text>
                      <Text fontWeight="bold">{generationStats.total_jobs || 0}</Text>
                    </Flex>
                    {generationStats.total_tokens > 0 && (
                      <Flex justify="space-between" color="gray.600" fontSize="sm" mt={1}>
                        <Text>Total Tokens</Text>
                        <Text fontWeight="bold">{generationStats.total_tokens?.toLocaleString()}</Text>
                      </Flex>
                    )}
                  </Box>
                ) : (
                  <Text color="gray.500" textAlign="center" py={8}>
                    No provider data yet. Start generating!
                  </Text>
                )}
              </CardBody>
            </Card>
          </SimpleGrid>
        </>
      )}

      {/* Recent Datasets */}
      <Card bg={cardBg} borderWidth="1px" borderRadius="lg" borderColor={borderColor} mb={8}>
        <CardHeader>
          <Heading size="md">Recent Datasets</Heading>
        </CardHeader>
        <Divider />
        <CardBody>
          {isLoading ? (
            <Text>Loading datasets...</Text>
          ) : recentDatasets.length > 0 ? (
            recentDatasets.map((dataset, index) => (
              <Box key={dataset.id} mb={index < recentDatasets.length - 1 ? 4 : 0}>
                <Flex justify="space-between" align="center">
                  <Flex align="center">
                    <Icon as={FiDatabase} mr={3} color="blue.500" />
                    <Box>
                      <Text fontWeight="medium">
                        <Link to={`/datasets/${dataset.id}`}>
                          {dataset.name}
                        </Link>
                      </Text>
                      <Text fontSize="sm" color="gray.600">
                        {dataset.domain} • {dataset.example_count} examples
                      </Text>
                    </Box>
                  </Flex>
                  <Box>
                    <Badge colorScheme={dataset.format === 'chat' ? 'blue' : 'green'}>
                      {dataset.format}
                    </Badge>
                  </Box>
                </Flex>
                {index < recentDatasets.length - 1 && <Divider mt={4} />}
              </Box>
            ))
          ) : (
            <Text>No datasets found. Create your first dataset!</Text>
          )}
        </CardBody>
      </Card>
    </Box>
  );
};

export default Dashboard;