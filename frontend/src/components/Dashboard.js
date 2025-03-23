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
import { FiDatabase, FiCpu, FiCheckCircle, FiActivity, FiArrowRight } from 'react-icons/fi';
import { useQuery } from '@tanstack/react-query';
import { getDatasets } from '../services/api';
import { Link } from 'react-router-dom';

const Dashboard = () => {
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  // Fetch datasets
  const { data: datasetsData, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => getDatasets()
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
  
  // Quick access cards
  const quickAccessCards = [
    {
      title: 'Generate Dataset',
      description: 'Create synthetic data across various domains',
      icon: FiCpu,
      color: 'blue',
      path: '/generator'
    },
    {
      title: 'Quality Control',
      description: 'Evaluate and improve dataset quality',
      icon: FiCheckCircle,
      color: 'green',
      path: '/quality'
    },
    {
      title: 'View Datasets',
      description: 'Browse, filter, and manage datasets',
      icon: FiDatabase,
      color: 'purple',
      path: '/datasets'
    },
    {
      title: 'Job Status',
      description: 'Monitor running and completed jobs',
      icon: FiActivity,
      color: 'orange',
      path: '/jobs'
    }
  ];
  
  return (
    <Box>
      <Heading size="lg" mb={6}>Dashboard</Heading>
      
      {/* Quick access cards */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        {quickAccessCards.map((card, index) => (
          <Card
            key={index}
            bg={cardBg}
            borderWidth="1px"
            borderRadius="lg"
            borderColor={borderColor}
            boxShadow="md"
            transition="all 0.2s"
            _hover={{ transform: 'translateY(-4px)', boxShadow: 'lg' }}
          >
            <CardBody>
              <Flex align="center" mb={3}>
                <Icon as={card.icon} fontSize="xl" color={`${card.color}.500`} mr={2} />
                <Heading size="md">{card.title}</Heading>
              </Flex>
              <Text color="gray.600" mb={4}>{card.description}</Text>
              <Button
                as={Link}
                to={card.path}
                colorScheme={card.color}
                size="sm"
                rightIcon={<FiArrowRight />}
              >
                Go
              </Button>
            </CardBody>
          </Card>
        ))}
      </SimpleGrid>
      
      {/* Stats overview */}
      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6} mb={8}>
        <Card bg={cardBg} borderWidth="1px" borderRadius="lg" borderColor={borderColor}>
          <CardBody>
            <Stat>
              <StatLabel>Total Datasets</StatLabel>
              <StatNumber>{stats.total}</StatNumber>
              <StatHelpText>
                {Object.entries(stats.formats).map(([format, count]) => (
                  <Badge key={format} mr={2} colorScheme={format === 'chat' ? 'blue' : 'green'}>
                    {format}: {count}
                  </Badge>
                ))}
              </StatHelpText>
            </Stat>
          </CardBody>
        </Card>
        
        <Card bg={cardBg} borderWidth="1px" borderRadius="lg" borderColor={borderColor}>
          <CardBody>
            <Stat>
              <StatLabel>Top Domains</StatLabel>
              <StatNumber>
                {Object.keys(stats.domains).length} domains
              </StatNumber>
              <StatHelpText>
                {Object.entries(stats.domains)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 3)
                  .map(([domain, count]) => (
                    <Badge key={domain} mr={2} colorScheme="purple">
                      {domain}: {count}
                    </Badge>
                  ))}
              </StatHelpText>
            </Stat>
          </CardBody>
        </Card>
        
        <Card bg={cardBg} borderWidth="1px" borderRadius="lg" borderColor={borderColor}>
          <CardBody>
            <Stat>
              <StatLabel>Current Status</StatLabel>
              <StatNumber>
                <Badge colorScheme="green">Ready</Badge>
              </StatNumber>
              <StatHelpText>
                All systems operational
              </StatHelpText>
            </Stat>
          </CardBody>
        </Card>
      </SimpleGrid>
      
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