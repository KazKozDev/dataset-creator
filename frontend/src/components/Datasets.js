import React, { useState } from 'react';
import {
  Box,
  Heading,
  Text,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Button,
  Badge,
  InputGroup,
  Input,
  InputLeftElement,
  Flex,
  Select,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Card,
  CardBody,
  Alert,
  AlertIcon,
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  FormControl,
  FormLabel,
  useDisclosure,
  useColorModeValue,
  VStack,
  HStack,
} from '@chakra-ui/react';
import {
  FiSearch,
  FiMoreVertical,
  FiDownload,
  FiTrash2,
  FiUpload,
  FiRefreshCw,
  FiTool,
  FiPlus,
  FiBarChart2,
} from 'react-icons/fi';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import { getDatasets, deleteDataset, uploadDataset, scanForDatasets } from '../services/api';

const Datasets = () => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { isOpen: isUploadOpen, onOpen: onUploadOpen, onClose: onUploadClose } = useDisclosure();

  const [searchQuery, setSearchQuery] = useState('');
  const [domainFilter, setDomainFilter] = useState('');
  const [formatFilter, setFormatFilter] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('DESC');
  const [dateFilter, setDateFilter] = useState('all'); // 'all', 'week', 'month', 'custom'
  const [customStartDate, setCustomStartDate] = useState('');
  const [customEndDate, setCustomEndDate] = useState('');

  const [uploadFile, setUploadFile] = useState(null);
  const [uploadName, setUploadName] = useState('');
  const [uploadDomain, setUploadDomain] = useState('unknown');
  const [uploadDescription, setUploadDescription] = useState('');

  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Fetch datasets
  const {
    data: datasetsData,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ['datasets', domainFilter, formatFilter, sortBy, sortOrder],
    queryFn: () => getDatasets({
      domain: domainFilter || undefined,
      format: formatFilter || undefined,
      sort_by: sortBy,
      sort_order: sortOrder,
    })
  });

  // Delete dataset mutation
  const deleteMutation = useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => {
      queryClient.invalidateQueries(['datasets']);
      toast({
        title: 'Dataset deleted',
        status: 'success',
        duration: 3000,
        isClosable: true
      });
    },
    onError: (error) => {
      toast({
        title: 'Error deleting dataset',
        description: error.response?.data?.detail || error.message,
        status: 'error',
        duration: 3000,
        isClosable: true
      });
    }
  });

  // Upload dataset mutation
  const uploadMutation = useMutation({
    mutationFn: (formData) => {
      const params = {
        name: uploadName || undefined,
        domain: uploadDomain || undefined,
        description: uploadDescription || undefined,
      };

      return uploadDataset(uploadFile, params);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries(['datasets']);
      onUploadClose();
      toast({
        title: 'Dataset uploaded',
        description: `Dataset "${data.name}" with ${data.example_count} examples`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });

      // Navigate to the new dataset
      navigate(`/datasets/${data.dataset_id}`);
    },
    onError: (err) => {
      toast({
        title: 'Failed to upload dataset',
        description: err.response?.data?.detail || 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    },
  });

  // Import existing file mutation
  const importMutation = useMutation({
    mutationFn: scanForDatasets,
    onSuccess: (data) => {
      if (data.files && data.files.length > 0) {
        toast({
          title: 'Files found',
          description: `Found ${data.files.length} JSONL files`,
          status: 'info',
          duration: 3000,
          isClosable: true,
        });
      } else {
        toast({
          title: 'No files found',
          description: 'No JSONL files were found in the scan',
          status: 'info',
          duration: 3000,
          isClosable: true,
        });
      }
    },
    onError: (err) => {
      toast({
        title: 'Failed to scan for files',
        description: err.response?.data?.detail || 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    },
  });

  // Filter datasets based on search query and date
  const filteredDatasets = datasetsData?.datasets
    ? datasetsData.datasets.filter((dataset) => {
      const searchLower = searchQuery.toLowerCase();

      // Search filter
      const matchesSearch = (
        dataset.name.toLowerCase().includes(searchLower) ||
        dataset.domain.toLowerCase().includes(searchLower) ||
        (dataset.description && dataset.description.toLowerCase().includes(searchLower))
      );

      // Date filter
      let matchesDate = true;
      if (dateFilter !== 'all') {
        const datasetDate = new Date(dataset.created_at);
        const now = new Date();

        if (dateFilter === 'week') {
          const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
          matchesDate = datasetDate >= weekAgo;
        } else if (dateFilter === 'month') {
          const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
          matchesDate = datasetDate >= monthAgo;
        } else if (dateFilter === 'custom') {
          if (customStartDate) {
            matchesDate = matchesDate && datasetDate >= new Date(customStartDate);
          }
          if (customEndDate) {
            matchesDate = matchesDate && datasetDate <= new Date(customEndDate);
          }
        }
      }

      return matchesSearch && matchesDate;
    })
    : [];

  // Extract unique domains for filter
  const uniqueDomains = datasetsData?.datasets
    ? [...new Set(datasetsData.datasets.map((d) => d.domain))]
    : [];

  const handleDeleteDataset = (id) => {
    if (window.confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
      deleteMutation.mutate(id);
    }
  };

  const handleUploadSubmit = (e) => {
    e.preventDefault();

    if (!uploadFile) {
      toast({
        title: 'No file selected',
        description: 'Please select a JSONL file to upload',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    uploadMutation.mutate();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadFile(file);

      // Auto-set name from filename if not provided
      if (!uploadName) {
        setUploadName(file.name.replace(/\.[^/.]+$/, ''));
      }
    }
  };

  return (
    <Box>
      <Flex justify="space-between" align="center" mb={6}>
        <Heading size="md">Datasets</Heading>
        <Flex>
          <Button
            leftIcon={<FiUpload />}
            colorScheme="blue"
            variant="outline"
            mr={2}
            onClick={onUploadOpen}
          >
            Upload
          </Button>
          <Button leftIcon={<FiPlus />} colorScheme="blue">
            Create New
          </Button>
        </Flex>
      </Flex>

      {/* Filters */}
      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={6} boxShadow="sm">
        <CardBody>
          <Flex direction={{ base: 'column', md: 'row' }} mb={4} gap={4}>
            {/* Search Input */}
            <InputGroup maxW={{ md: '300px' }}>
              <InputLeftElement pointerEvents="none">
                <FiSearch color="gray.300" />
              </InputLeftElement>
              <Input
                placeholder="Search datasets..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </InputGroup>

            {/* Domain Filter */}
            <Select
              placeholder="All Domains"
              value={domainFilter}
              onChange={(e) => setDomainFilter(e.target.value)}
              maxW={{ md: '200px' }}
            >
              {uniqueDomains.map((domain) => (
                <option key={domain} value={domain}>
                  {domain}
                </option>
              ))}
            </Select>

            {/* Format Filter */}
            <Select
              placeholder="All Formats"
              value={formatFilter}
              onChange={(e) => setFormatFilter(e.target.value)}
              maxW={{ md: '200px' }}
            >
              <option value="chat">Chat</option>
              <option value="instruction">Instruction</option>
            </Select>

            {/* Date Filter */}
            <VStack align="stretch" spacing={2}>
              <HStack spacing={2}>
                <Button
                  size="sm"
                  variant={dateFilter === 'all' ? 'solid' : 'outline'}
                  colorScheme={dateFilter === 'all' ? 'blue' : 'gray'}
                  onClick={() => setDateFilter('all')}
                >
                  All Time
                </Button>
                <Button
                  size="sm"
                  variant={dateFilter === 'week' ? 'solid' : 'outline'}
                  colorScheme={dateFilter === 'week' ? 'blue' : 'gray'}
                  onClick={() => setDateFilter('week')}
                >
                  Last 7 Days
                </Button>
                <Button
                  size="sm"
                  variant={dateFilter === 'month' ? 'solid' : 'outline'}
                  colorScheme={dateFilter === 'month' ? 'blue' : 'gray'}
                  onClick={() => setDateFilter('month')}
                >
                  Last 30 Days
                </Button>
                <Button
                  size="sm"
                  variant={dateFilter === 'custom' ? 'solid' : 'outline'}
                  colorScheme={dateFilter === 'custom' ? 'blue' : 'gray'}
                  onClick={() => setDateFilter('custom')}
                >
                  Custom Range
                </Button>
              </HStack>

              {dateFilter === 'custom' && (
                <HStack spacing={2}>
                  <Input
                    type="date"
                    size="sm"
                    value={customStartDate}
                    onChange={(e) => setCustomStartDate(e.target.value)}
                    placeholder="Start date"
                  />
                  <Text fontSize="sm">to</Text>
                  <Input
                    type="date"
                    size="sm"
                    value={customEndDate}
                    onChange={(e) => setCustomEndDate(e.target.value)}
                    placeholder="End date"
                  />
                </HStack>
              )}
            </VStack>

            {/* Sort Options */}
            <Flex align="center">
              <Text mr={2} whiteSpace="nowrap">
                Sort by:
              </Text>
              <Select
                value={`${sortBy}_${sortOrder}`}
                onChange={(e) => {
                  const [newSortBy, newSortOrder] = e.target.value.split('_');
                  setSortBy(newSortBy);
                  setSortOrder(newSortOrder);
                }}
                maxW={{ md: '200px' }}
              >
                <option value="created_at_DESC">Date (Newest)</option>
                <option value="created_at_ASC">Date (Oldest)</option>
                <option value="name_ASC">Name (A-Z)</option>
                <option value="name_DESC">Name (Z-A)</option>
                <option value="example_count_DESC">Size (Largest)</option>
                <option value="example_count_ASC">Size (Smallest)</option>
              </Select>
            </Flex>
          </Flex>

          <Flex justifyContent="space-between" alignItems="center">
            <Text color="gray.600" fontSize="sm">
              {filteredDatasets.length} datasets found
            </Text>

            <Flex>
              <Button
                size="sm"
                variant="ghost"
                leftIcon={<FiRefreshCw />}
                onClick={() => refetch()}
                isLoading={isLoading}
                mr={2}
              >
                Refresh
              </Button>

              <Button
                size="sm"
                variant="ghost"
                leftIcon={<FiTool />}
                onClick={() => importMutation.mutate()}
                isLoading={importMutation.isLoading}
              >
                Scan Files
              </Button>
            </Flex>
          </Flex>
        </CardBody>
      </Card>

      {/* Error state */}
      {isError && (
        <Alert status="error" mb={6}>
          <AlertIcon />
          {error.response?.data?.detail || 'Failed to load datasets. Please try again.'}
        </Alert>
      )}

      {/* Datasets Table */}
      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} boxShadow="sm">
        <Box overflowX="auto">
          <Table variant="simple">
            <Thead>
              <Tr>
                <Th width="30%">Name</Th>
                <Th width="20%" textAlign="center">Domain</Th>
                <Th width="15%" textAlign="center">Format</Th>
                <Th width="15%" textAlign="center">Examples</Th>
                <Th width="20%" textAlign="center">Created</Th>
                <Th width="80px"></Th>
              </Tr>
            </Thead>
            <Tbody>
              {isLoading ? (
                <Tr>
                  <Td colSpan={6} textAlign="center" py={10}>
                    Loading datasets...
                  </Td>
                </Tr>
              ) : filteredDatasets.length === 0 ? (
                <Tr>
                  <Td colSpan={6} textAlign="center" py={10}>
                    <VStack spacing={4}>
                      <Text fontSize="lg" fontWeight="medium" color="gray.600">
                        {searchQuery || domainFilter || formatFilter
                          ? 'No datasets match your filters'
                          : 'No datasets found'}
                      </Text>
                      <Text fontSize="sm" color="gray.500">
                        {searchQuery || domainFilter || formatFilter
                          ? 'Try adjusting your search or filters'
                          : 'Get started by creating or uploading a dataset'}
                      </Text>
                      {!searchQuery && !domainFilter && !formatFilter && (
                        <HStack spacing={3}>
                          <Button
                            leftIcon={<FiPlus />}
                            colorScheme="blue"
                            size="sm"
                            onClick={() => navigate('/generator')}
                          >
                            Generate Dataset
                          </Button>
                          <Button
                            leftIcon={<FiUpload />}
                            variant="outline"
                            size="sm"
                            onClick={onUploadOpen}
                          >
                            Upload Dataset
                          </Button>
                        </HStack>
                      )}
                    </VStack>
                  </Td>
                </Tr>
              ) : (
                filteredDatasets.map((dataset) => (
                  <Tr
                    key={dataset.id}
                    _hover={{ bg: 'gray.50' }}
                    transition="background-color 0.2s"
                  >
                    <Td>
                      <Box>
                        <Text
                          fontWeight="medium"
                          cursor="pointer"
                          onClick={() => navigate(`/datasets/${dataset.id}`)}
                          _hover={{ color: 'blue.500' }}
                        >
                          {dataset.name}
                        </Text>
                        {dataset.description && (
                          <Text
                            fontSize="sm"
                            color="gray.600"
                            noOfLines={1}
                            pl={1}
                          >
                            {dataset.description}
                          </Text>
                        )}
                      </Box>
                    </Td>
                    <Td textAlign="center">{dataset.domain}</Td>
                    <Td textAlign="center">
                      <Badge colorScheme={dataset.format === 'chat' ? 'blue' : 'green'}>
                        {dataset.format}
                      </Badge>
                    </Td>
                    <Td textAlign="center">{dataset.example_count}</Td>
                    <Td textAlign="center">
                      {new Date(dataset.created_at).toLocaleString()}
                    </Td>
                    <Td textAlign="center">
                      <Menu>
                        <MenuButton
                          as={IconButton}
                          icon={<FiMoreVertical />}
                          variant="ghost"
                          size="sm"
                          aria-label="Dataset options"
                        />
                        <MenuList>
                          <MenuItem icon={<FiBarChart2 />} as={Link} to={`/quality?dataset=${dataset.id}`}>
                            Quality Check
                          </MenuItem>
                          <MenuItem icon={<FiDownload />} as={Link} to={`/datasets/${dataset.id}?export=true`}>
                            Export
                          </MenuItem>
                          <MenuItem
                            icon={<FiTrash2 />}
                            color="red.500"
                            onClick={() => handleDeleteDataset(dataset.id)}
                          >
                            Delete
                          </MenuItem>
                        </MenuList>
                      </Menu>
                    </Td>
                  </Tr>
                ))
              )}
            </Tbody>
          </Table>
        </Box>
      </Card>

      {/* Upload Modal */}
      <Modal isOpen={isUploadOpen} onClose={onUploadClose}>
        <ModalOverlay />
        <ModalContent>
          <form onSubmit={handleUploadSubmit}>
            <ModalHeader>Upload Dataset</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <FormControl mb={4}>
                <FormLabel>JSONL File</FormLabel>
                <Input type="file" accept=".jsonl" onChange={handleFileChange} />
                <Text fontSize="sm" color="gray.600" mt={1}>
                  Only JSONL files in chat or instruction format are supported
                </Text>
              </FormControl>

              <FormControl mb={4}>
                <FormLabel>Dataset Name</FormLabel>
                <Input
                  placeholder="My Dataset"
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                />
              </FormControl>

              <FormControl mb={4}>
                <FormLabel>Domain</FormLabel>
                <Select
                  placeholder="Select domain"
                  value={uploadDomain}
                  onChange={(e) => setUploadDomain(e.target.value)}
                >
                  {uniqueDomains.map((domain) => (
                    <option key={domain} value={domain}>
                      {domain}
                    </option>
                  ))}
                  <option value="unknown">Unknown</option>
                </Select>
              </FormControl>

              <FormControl>
                <FormLabel>Description</FormLabel>
                <Input
                  placeholder="Optional description"
                  value={uploadDescription}
                  onChange={(e) => setUploadDescription(e.target.value)}
                />
              </FormControl>
            </ModalBody>

            <ModalFooter>
              <Button variant="ghost" mr={3} onClick={onUploadClose}>
                Cancel
              </Button>
              <Button
                colorScheme="blue"
                type="submit"
                isLoading={uploadMutation.isLoading}
              >
                Upload
              </Button>
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default Datasets;